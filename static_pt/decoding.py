import torch
import torch.nn.functional as F


def _gather_beams(x, beam_indices):
  """
  Efficiently gathers beam data across a batch during the selection step.

  This utility uses one-hot contraction (via torch.gather) to select the top-M
  sequences from the candidate pool while preserving batch and history dimensions.
  It is designed to be highly efficient on hardware accelerators by avoiding
  explicit Python loops.

  Args:
      x (torch.Tensor): The source tensor to gather from, usually of shape
          (batch_size, old_beam_size, ...).
      beam_indices (torch.Tensor): The indices of the beams to select, of shape
          (batch_size, new_beam_size).

  Returns:
      torch.Tensor: The gathered tensor of shape (batch_size, new_beam_size, ...).
  """
  batch_size, new_beam_size = beam_indices.shape
  view_shape = [batch_size, new_beam_size] + [1] * (x.dim() - 2)
  expand_shape = [batch_size, new_beam_size] + list(x.shape[2:])
  indices = beam_indices.view(view_shape).expand(expand_shape)
  return x.gather(1, indices)


@torch.inference_mode()
def generate_and_apply_logprobs_mask(
    flat_logprobs, flat_states, packed_csr, csr_indptr,
    limit, vocab_size, device
):
  """
  Performs vectorized sparse candidate extraction from the STATIC CSR matrix.

  This kernel replaces irregular "pointer-chasing" trie traversals with a single
  vectorized burst-read. It retrieves the log-probabilities for all valid child
  tokens of the current trie states in one coalesced operation, ensuring $O(1)$
  latency relative to the total constraint set size.

  Args:
      flat_logprobs (torch.Tensor): Model-predicted log-probabilities of shape
          (batch_size * beam_size, vocab_size).
      flat_states (torch.Tensor): Current trie state IDs for each beam.
      packed_csr (torch.Tensor): The flat transition table [Token ID, Next State ID].
      csr_indptr (torch.Tensor): The CSR row pointer array identifying segments.
      limit (int): The maximum branching factor (K) for the current trie depth.
      vocab_size (int): The total token vocabulary size.
      device (torch.device): The accelerator device (CUDA/TPU).

  Returns:
      tuple: A tuple containing:
          - candidate_logprobs (Tensor): Log-probs for valid children, shape (B*M, K).
          - dense_indices (Tensor): Token IDs for valid children, shape (B*M, K).
          - dense_data (Tensor): Next state IDs for valid children, shape (B*M, K).
  """
  # 1. Fetch Sparse Rows (Burst Read)
  # We perform a coalesced read from the CSR by indexing into the row pointers.
  starts = csr_indptr[flat_states.long()]
  actual_lens = csr_indptr[flat_states.long() + 1] - starts

  # Create a grid of offsets to gather exactly 'limit' (K) candidates per state.
  offsets = torch.arange(limit, device=device)
  gather_indices = starts.unsqueeze(1) + offsets.unsqueeze(0)

  # Clamp indices to handle states with fewer than 'limit' children safely.
  max_idx = packed_csr.size(0) - 1
  safe_gather_indices = gather_indices.clamp(max=max_idx)

  # Retrieve [Token, NextState] pairs directly from High-Bandwidth Memory (HBM).
  gathered_vals = packed_csr[safe_gather_indices]
  dense_indices = gathered_vals[..., 0]
  dense_data = gathered_vals[..., 1]

  # 2. Validity Masking
  # Mask out 'padding' slots if a trie node has fewer than 'limit' children.
  valid_mask = offsets.unsqueeze(0) < actual_lens.unsqueeze(1)

  # 3. Logprob Gathering
  # Gather only the specific log-probabilities corresponding to valid tokens.
  safe_dense_indices = dense_indices.long().clamp(max=vocab_size - 1)
  candidate_logprobs = flat_logprobs.gather(1, safe_dense_indices)

  # Apply -inf mask to invalidate non-existent paths in the prefix tree.
  candidate_logprobs = torch.where(
      valid_mask,
      candidate_logprobs,
      torch.tensor(-float('inf'), device=device)
  )

  return candidate_logprobs, dense_indices, dense_data


@torch.inference_mode()
def sparse_transition_packed_torch(
    batch_size, beam_size, tokens_per_beam, pad_token, max_sample_len, vocab_size,
    max_branch_factors, packed_csr, csr_indptr, start_mask, dense_mask, dense_states,
    device, d_dense=2
):
  """
  Main harness for STATIC constrained beam search.

  This function executes the full autoregressive decoding loop. It uses a hybrid
  approach, specializing the first 'd_dense' codewords into dense lookup tables
  to maximize throughput on "hot" paths, and using a CSR matrix for the
  high-cardinality "sparse tail".

  Args:
      batch_size (int): Number of sequences to decode in parallel.
      beam_size (int): Number of beams to maintain per sequence.
      tokens_per_beam (int): Number of candidate tokens to consider per beam.
      pad_token (int): Token ID used for padding.
      max_sample_len (int): Length (L) of the Semantic IDs being decoded.
      vocab_size (int): Size of the token vocabulary.
      max_branch_factors (tuple): Maximum branching factors per level.
      packed_csr (torch.Tensor): Flattened trie transitions.
      csr_indptr (torch.Tensor): CSR row pointers.
      start_mask (torch.Tensor): 1D validity mask for the root (Level 0).
      dense_mask (torch.Tensor): d-dimensional dense validity mask.
      dense_states (torch.Tensor): d-dimensional dense state table.
      device (torch.device): Device to execute on.
      d_dense (int): Number of initial dense layers.
          NOTE: In practice, we only support d=1 and d=2 (recommended).
          Users requiring deeper specialization (d >= 3) must implement robust
          multi-dimensional broadcasting for those specific steps.

  Returns:
      torch.Tensor: The decoded token sequences of shape (batch_size, beam_size, L).
  """
  # --- 1. INITIAL STEP (Codeword 1) ---
  # Apply the root mask to restrict the first token to valid starting codewords.
  initial_logits = torch.rand(batch_size, 1, vocab_size, device=device)
  raw_logprobs = F.log_softmax(initial_logits[:, 0, :], dim=-1)

  initial_logprobs = torch.where(
      start_mask,
      raw_logprobs,
      torch.tensor(-float('inf'), device=device)
  )
  top_logprobs, top_tokens = torch.topk(initial_logprobs, beam_size, dim=-1)

  # Initialize decoding buffers.
  token_buffer = torch.full(
      (batch_size, beam_size, max_sample_len),
      pad_token,
      dtype=top_tokens.dtype,
      device=device
  )
  token_buffer[:, :, 0] = top_tokens

  # Map Level-0 tokens to their initial trie state IDs (Token T -> ID T+1).
  current_transition_states = top_tokens + 1
  current_token_scores = top_logprobs

  # --- 2. AUTOREGRESSIVE LOOP (Codewords 2 to L) ---
  for step in range(max_sample_len - 1):
    # Generate next-token logits from the model.
    flat_logits = torch.rand(batch_size * beam_size, 1, vocab_size, device=device)
    flat_logprobs = F.log_softmax(flat_logits[:, 0, :], dim=-1)
    flat_states = current_transition_states.view(batch_size * beam_size)

    # Apply hybrid dense/sparse masking.
    if step < d_dense - 1:
      # --- DENSE SPECIALIZATION ---
      parent_tokens = (flat_states - 1).long()
      masks = dense_mask[parent_tokens]

      flat_logprobs = torch.where(
          masks,
          flat_logprobs,
          torch.tensor(-float('inf'), device=device)
      )

      topk_logprobs, topk_indices = torch.topk(flat_logprobs, tokens_per_beam, dim=-1)
      next_state_candidates = dense_states[parent_tokens.unsqueeze(1), topk_indices.long()]

      limit = tokens_per_beam
      candidates_logprobs, candidates_indices, candidates_states = (
          topk_logprobs, topk_indices, next_state_candidates
      )
    else:
      # --- SPARSE CSR LOOKUP ---
      # Transition to CSR logic once the state space becomes too sparse for dense tables.
      limit = max_branch_factors[step + 1]
      candidates_logprobs, candidates_indices, candidates_states = generate_and_apply_logprobs_mask(
          flat_logprobs, flat_states, packed_csr, csr_indptr,
          limit, vocab_size, device
      )

    # --- SCORE & BEAM UPDATE ---
    # Update cumulative log-probability scores.
    scores = current_token_scores.unsqueeze(2) + candidates_logprobs.view(batch_size, beam_size, limit)
    flat_scores = scores.view(batch_size, beam_size * limit)

    # Select the top beams for the next step.
    top_scores, flat_top_indices = torch.topk(flat_scores, beam_size, dim=-1)

    # Recover token IDs and state transitions for the selected beams.
    top_beam_indices = flat_top_indices // limit
    flat_tokens = candidates_indices.view(batch_size, beam_size * limit)
    flat_next_states = candidates_states.view(batch_size, beam_size * limit)

    top_tokens = _gather_beams(flat_tokens, flat_top_indices)
    current_transition_states = _gather_beams(flat_next_states, flat_top_indices)

    # Update history and scores.
    token_buffer = _gather_beams(token_buffer, top_beam_indices)
    token_buffer[:, :, step + 1] = top_tokens
    current_token_scores = top_scores

  return token_buffer
