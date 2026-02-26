# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_beams(x: torch.Tensor, beam_indices: torch.Tensor) -> torch.Tensor:
  """Efficiently gathers beam data across a batch during the selection step.

  Uses torch.gather to select the top-M sequences from the candidate pool
  while preserving batch and history dimensions.

  Args:
      x: The source tensor to gather from.
          Shape: (batch_size, old_beam_size, ...).
      beam_indices: The indices of the beams to select.
          Shape: (batch_size, new_beam_size).

  Returns:
      The gathered tensor.
          Shape: (batch_size, new_beam_size, ...).
  """
  batch_size, new_beam_size = beam_indices.shape
  view_shape = [batch_size, new_beam_size] + [1] * (x.dim() - 2)
  expand_shape = [batch_size, new_beam_size] + list(x.shape[2:])
  indices = beam_indices.view(view_shape).expand(expand_shape)
  return x.gather(1, indices)


@torch.inference_mode()
def generate_and_apply_logprobs_mask(
    flat_logprobs: torch.Tensor,
    flat_states: torch.Tensor,
    packed_csr: torch.Tensor,
    csr_indptr: torch.Tensor,
    limit: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Performs vectorized sparse candidate extraction from the STATIC CSR matrix.

  This kernel replaces irregular "pointer-chasing" trie traversals with a single
  vectorized burst-read. It retrieves the log-probabilities for all valid child
  tokens of the current trie states in one coalesced operation, ensuring $O(1)$
  latency relative to the total constraint set size.

  Args:
      flat_logprobs: Model-predicted log-probabilities.
          Shape: (batch_size * beam_size, vocab_size).
      flat_states: Current trie state IDs for each beam.
          Shape: (batch_size * beam_size,).
      packed_csr: The flat transition table [Token ID, Next State ID].
          Shape: (num_transitions + V, 2).
      csr_indptr: The CSR row pointer array identifying segments.
          Shape: (num_states + 2,).
      limit: The maximum branching factor (K) for the current trie depth.
      vocab_size: The total token vocabulary size (V).
      device: The accelerator device (CUDA/TPU).

  Returns:
      A tuple containing:
          - candidate_logprobs: Log-probs for valid children.
              Shape: (batch_size * beam_size, K).
          - candidate_token_ids: Token IDs for valid children.
              Shape: (batch_size * beam_size, K).
          - candidate_next_states: Next state IDs for valid children.
              Shape: (batch_size * beam_size, K).
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
  candidate_token_ids = gathered_vals[..., 0]
  candidate_next_states = gathered_vals[..., 1]

  # 2. Validity Masking
  # Mask out 'padding' slots if a trie node has fewer than 'limit' children.
  valid_mask = offsets.unsqueeze(0) < actual_lens.unsqueeze(1)

  # 3. Logprob Gathering
  # Gather only the specific log-probabilities corresponding to valid tokens.
  safe_token_ids = candidate_token_ids.long().clamp(max=vocab_size - 1)
  candidate_logprobs = flat_logprobs.gather(1, safe_token_ids)

  # Apply -inf mask to invalidate non-existent paths in the prefix tree.
  candidate_logprobs = torch.where(
      valid_mask, candidate_logprobs, torch.tensor(-float('inf'), device=device)
  )

  return candidate_logprobs, candidate_token_ids, candidate_next_states


class RandomModel(nn.Module):
  """A dummy model that acts like a Transformer but outputs random logits.

  Used to benchmark the throughput of the decoding harness without the
  computational overhead of a real neural network.
  """

  def __init__(self, vocab_size: int, device: torch.device):
    super().__init__()
    self.vocab_size = vocab_size
    self.device = device
    self.to(device)

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    """Generates random logits for the next token prediction.

    Args:
        input_ids: Shape (batch_size, seq_len).

    Returns:
        Random logits. Shape: (batch_size, 1, vocab_size).
    """
    batch_size = input_ids.size(0)
    return torch.rand(batch_size, 1, self.vocab_size, device=self.device)


@torch.inference_mode()
def sparse_transition_torch(
    model: nn.Module,
    batch_size: int,
    beam_size: int,
    tokens_per_beam: int,
    start_token: int,
    max_sample_len: int,
    vocab_size: int,
    max_branch_factors: tuple[int, ...],
    packed_csr: torch.Tensor,
    csr_indptr: torch.Tensor,
    start_mask: torch.Tensor,
    dense_mask: torch.Tensor,
    dense_states: torch.Tensor,
    device: torch.device,
    d_dense: int = 2,
) -> torch.Tensor:
  """Main harness for STATIC constrained beam search using a PyTorch model.

  Executes the full autoregressive decoding loop. Generates logits from the
  provided `model` and applies the STATIC hybrid masking strategy
  (Dense + CSR) to strictly enforce the constraint graph.

  Args:
      model: A PyTorch model that accepts input_ids of shape (B*M, 1) and
          returns logits of shape (B*M, 1, V).
      batch_size: Number of sequences to decode in parallel (B).
      beam_size: Number of beams to maintain per sequence (M).
      tokens_per_beam: Number of candidate tokens to consider per beam.
      start_token: The token ID used to initiate decoding (e.g., BOS or PAD).
      max_sample_len: Length (L) of the Semantic IDs being decoded.
      vocab_size: Size of the token vocabulary (V).
      max_branch_factors: Maximum branching factors per level.
          Length: L.
      packed_csr: Flattened trie transitions (Sparse Tail).
          Shape: (num_transitions + V, 2).
      csr_indptr: CSR row pointers.
          Shape: (num_states + 2,).
      start_mask: 1D validity mask for the root (Level 0).
          Shape: (V,).
      dense_mask: d_dense-dimensional dense validity mask (Hot Head).
          Shape: (V,) * d_dense.
      dense_states: d_dense-dimensional dense state table.
          Shape: (V,) * d_dense.
      device: Device to execute on.
      d_dense: Number of initial dense layers.
          NOTE: In practice, we only support d_dense=1 and d_dense=2
          (recommended).

  Returns:
      The decoded token sequences.
          Shape: (batch_size, beam_size, max_sample_len).
  """
  # --- 1. INITIAL STEP (Codeword 1) ---
  # Use the specific start_token expected by the model (BOS/PAD)
  initial_input = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

  # Get real logits from the model
  initial_logits = model(initial_input)
  raw_logprobs = F.log_softmax(initial_logits[:, 0, :], dim=-1)

  # Apply the root mask to restrict the first token
  initial_logprobs = torch.where(
      start_mask, raw_logprobs, torch.tensor(-float('inf'), device=device)
  )
  top_logprobs, top_tokens = torch.topk(initial_logprobs, beam_size, dim=-1)

  # Initialize decoding buffers
  token_buffer = torch.full(
      (batch_size, beam_size, max_sample_len),
      start_token,
      dtype=top_tokens.dtype,
      device=device,
  )
  token_buffer[:, :, 0] = top_tokens

  # Map Level-0 tokens to their initial trie state IDs (Token T -> ID T+1)
  current_transition_states = top_tokens + 1
  current_token_scores = top_logprobs

  # --- 2. AUTOREGRESSIVE LOOP (Codewords 2 to L) ---
  for step in range(max_sample_len - 1):
    # Prepare input: Flatten the top tokens from the previous step
    # Shape: (batch_size * beam_size, 1)
    flat_input_ids = top_tokens.view(batch_size * beam_size, 1)

    # Generate next-token logits from the model
    flat_logits = model(flat_input_ids)
    flat_logprobs = F.log_softmax(flat_logits[:, 0, :], dim=-1)

    flat_states = current_transition_states.view(batch_size * beam_size)

    # Apply hybrid dense/sparse masking
    if step < d_dense - 1:
      # --- DENSE SPECIALIZATION ---
      # Reconstruct previous token from state ID (Valid for d_dense=2)
      parent_tokens = (flat_states - 1).long()
      masks = dense_mask[parent_tokens]

      flat_logprobs = torch.where(
          masks, flat_logprobs, torch.tensor(-float('inf'), device=device)
      )

      topk_logprobs, topk_indices = torch.topk(
          flat_logprobs, tokens_per_beam, dim=-1
      )

      # Map winners to next trie states using dense table
      next_state_candidates = dense_states[
          parent_tokens.unsqueeze(1), topk_indices.long()
      ]

      limit = tokens_per_beam
      candidates_logprobs, candidates_indices, candidates_states = (
          topk_logprobs,
          topk_indices,
          next_state_candidates,
      )
    else:
      # --- SPARSE CSR LOOKUP ---
      # Transition to CSR logic once the state space becomes too sparse
      limit = max_branch_factors[step + 1]
      candidates_logprobs, candidates_indices, candidates_states = (
          generate_and_apply_logprobs_mask(
              flat_logprobs,
              flat_states,
              packed_csr,
              csr_indptr,
              limit,
              vocab_size,
              device,
          )
      )

    # --- SCORE & BEAM UPDATE ---
    scores = current_token_scores.unsqueeze(2) + candidates_logprobs.view(
        batch_size, beam_size, limit
    )
    flat_scores = scores.view(batch_size, beam_size * limit)

    # Select the top beams for the next step
    top_scores, flat_top_indices = torch.topk(flat_scores, beam_size, dim=-1)

    # Recover token IDs and state transitions for the selected beams
    top_beam_indices = flat_top_indices // limit
    flat_tokens = candidates_indices.view(batch_size, beam_size * limit)
    flat_next_states = candidates_states.view(batch_size, beam_size * limit)

    top_tokens = _gather_beams(flat_tokens, flat_top_indices)
    current_transition_states = _gather_beams(
        flat_next_states, flat_top_indices
    )

    # Update history and scores
    token_buffer = _gather_beams(token_buffer, top_beam_indices)
    token_buffer[:, :, step + 1] = top_tokens
    current_token_scores = top_scores

  return token_buffer
