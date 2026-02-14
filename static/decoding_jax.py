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

import functools
import jax
from jax import lax
import jax.numpy as jnp


def get_probs(shape, key):
  """Generates mock logits for testing and benchmarking.

  NOTE: This function generates probabilities randomly using a uniform
  distribution rather than using a real trained model. It is intended
  strictly for performance profiling and unit testing.

  Args:
      shape (tuple): The desired output shape, e.g., (batch_size, 1,
        vocab_size).
      key (jax.random.PRNGKey): JAX PRNG key for random number generation.

  Returns:
      tuple: A tuple containing:
          - logits (jnp.ndarray): The randomly generated logits.
          - next_key (jax.random.PRNGKey): The next PRNG key for
          reproducibility.
  """
  key, subkey = jax.random.split(key)
  return jax.random.uniform(subkey, shape, minval=0, maxval=1), key


def _gather_beams(x, beam_indices):
  """Efficiently gathers beam data across a batch during the selection step.

  This utility uses one-hot contraction for TPU efficiency to select the top-M
  sequences from the candidate pool while preserving batch and history
  dimensions.
  It is designed to be highly efficient on hardware accelerators by avoiding
  explicit Python loops.

  Args:
      x (jnp.ndarray): The source tensor to gather from, usually of shape
        (batch_size, old_beam_size, ...).
      beam_indices (jnp.ndarray): The indices of the beams to select, of shape
        (batch_size, new_beam_size).

  Returns:
      jnp.ndarray: The gathered tensor of shape (batch_size, new_beam_size,
      ...).
  """
  # Extract dimensions from tensor shapes
  _, old_beam_size = x.shape[:2]

  # Create one-hot mask for the selection
  oh_beam_indices = jax.nn.one_hot(beam_indices, old_beam_size, dtype=jnp.int32)
  # Perform one-hot contraction via einsum
  return jnp.einsum("beo,bo...->be...", oh_beam_indices, x).astype(x.dtype)


def generate_and_apply_logprobs_mask(
    flat_logprobs, flat_states, packed_csr, csr_indptr, limit, vocab_size
):
  """Performs vectorized sparse candidate extraction from the STATIC CSR matrix.

  This kernel replaces irregular "pointer-chasing" trie traversals with a single
  vectorized burst-read. It retrieves the log-probabilities for all valid child
  tokens of the current trie states in one coalesced operation, ensuring $O(1)$
  latency relative to the total constraint set size.

  Args:
      flat_logprobs (jnp.ndarray): Model-predicted log-probabilities of shape
        (batch_size * beam_size, vocab_size).
      flat_states (jnp.ndarray): Current trie state IDs for each beam.
      packed_csr (jnp.ndarray): The flat transition table [Token ID, Next State
        ID].
      csr_indptr (jnp.ndarray): The CSR row pointer array identifying segments.
      limit (int): The maximum branching factor (K) for the current trie depth.
      vocab_size (int): The total token vocabulary size.

  Returns:
      tuple: A tuple containing:
          - candidate_logprobs (jnp.ndarray): Log-probs for valid children,
          shape (B*M, K).
          - dense_indices (jnp.ndarray): Token IDs for valid children, shape
          (B*M, K).
          - dense_data (jnp.ndarray): Next state IDs for valid children, shape
          (B*M, K).
  """
  # 1. Fetch Sparse Rows (Burst Read)
  starts = csr_indptr[flat_states]
  actual_lens = csr_indptr[flat_states + 1] - starts

  # Create a grid of offsets to gather exactly 'limit' (K) candidates per state.
  offsets = jnp.arange(limit)
  # Broadcast: (Batch*Beam, 1) + (1, Limit) -> (Batch*Beam, Limit)
  gather_indices = starts[:, None] + offsets[None, :]

  # 2. Gather Data from CSR
  gathered_vals = jnp.take(
      packed_csr, gather_indices, axis=0, mode="fill", fill_value=0
  )
  dense_indices = gathered_vals[:, :, 0]  # Token IDs
  dense_data = gathered_vals[:, :, 1]  # Next States

  # 3. Logprob Gathering
  # Clamp indices to ensure we never gather from 'vocab_size' (which is OOB).
  # We use (vocab_size - 1) as a safe dummy index.
  safe_dense_indices = jnp.clip(dense_indices, max=vocab_size - 1)
  candidate_logprobs = jnp.take_along_axis(
      flat_logprobs, safe_dense_indices, axis=1
  )

  # 4. Validity Masking
  valid_mask = offsets[None, :] < actual_lens[:, None]
  safe_logprobs = jnp.where(valid_mask, candidate_logprobs, -jnp.inf)

  return safe_logprobs, dense_indices, dense_data


@functools.partial(
    jax.jit,
    static_argnames=(
        "batch_size",
        "beam_size",
        "tokens_per_beam",
        "pad_token",
        "max_sample_len",
        "vocab_size",
        "max_branch_factors",
        "d_dense",
    ),
)
def sparse_transition_packed_jit(
    key,
    batch_size,
    beam_size,
    tokens_per_beam,
    pad_token,
    max_sample_len,
    vocab_size,
    max_branch_factors,
    packed_csr,
    csr_indptr,
    start_mask,
    dense_mask,
    dense_states,
    d_dense=2,
):
  """Main harness for STATIC constrained beam search.

  This function executes the full autoregressive decoding loop. It uses a hybrid
  approach, specializing the first 'd_dense' codewords into dense lookup tables
  to maximize throughput on "hot" paths, and using a CSR matrix for the
  high-cardinality "sparse tail".

  Args:
      key (jax.random.PRNGKey): PRNG key for generating mock logits.
      batch_size (int): Number of sequences to decode in parallel.
      beam_size (int): Number of beams to maintain per sequence.
      tokens_per_beam (int): Number of candidate tokens to consider per beam.
      pad_token (int): Token ID used for padding.
      max_sample_len (int): Length (L) of the Semantic IDs being decoded.
      vocab_size (int): Size of the token vocabulary.
      max_branch_factors (tuple): Maximum branching factors per level.
      packed_csr (jnp.ndarray): Flattened trie transitions.
      csr_indptr (jnp.ndarray): CSR row pointers.
      start_mask (jnp.ndarray): 1D validity mask for the root (Level 0).
      dense_mask (jnp.ndarray): d-dimensional dense validity mask.
      dense_states (jnp.ndarray): d-dimensional dense state table.
      d_dense (int): Number of initial dense layers.
          NOTE: In practice, we only support d=1 and d=2 (recommended). Users
            requiring deeper specialization (d >= 3) must implement robust
            multi-dimensional broadcasting for those specific steps.

  Returns:
      jnp.ndarray: The decoded token sequences of shape (batch_size, beam_size,
      L).
  """
  # --- 1. INITIAL STEP (Codeword 1) ---
  # Apply the root mask to restrict the first token to valid starting codewords.
  initial_logits, key = get_probs((batch_size, 1, vocab_size), key)
  raw_logprobs = jax.nn.log_softmax(initial_logits[:, 0, :])

  # Root mask is a boolean array of shape (vocab_size,)
  initial_logprobs = jnp.where(start_mask, raw_logprobs, -jnp.inf)
  top_logprobs, top_tokens = lax.top_k(initial_logprobs, beam_size)

  # Initialize decoding buffers.
  token_buffer = jnp.full(
      (batch_size, beam_size, max_sample_len), pad_token, dtype=top_tokens.dtype
  )
  token_buffer = token_buffer.at[:, :, 0].set(top_tokens)

  # Map Level-0 tokens to their initial trie state IDs (Token T -> ID T+1).
  current_transition_states = top_tokens + 1
  current_token_scores = top_logprobs

  # --- 2. AUTOREGRESSIVE LOOP (Codewords 2 to L) ---
  for step in range(max_sample_len - 1):
    # Generate next-token logits from the model.
    flat_logits, key = get_probs((batch_size * beam_size, 1, vocab_size), key)
    flat_logprobs = jax.nn.log_softmax(flat_logits[:, 0, :])
    flat_states = current_transition_states.reshape(batch_size * beam_size)

    # Apply hybrid dense/sparse masking.
    if step < d_dense - 1:
      # --- DENSE SPECIALIZATION ---
      parent_tokens = (flat_states - 1).astype(jnp.int32)
      masks = dense_mask[parent_tokens]

      flat_logprobs = jnp.where(masks, flat_logprobs, -jnp.inf)

      topk_logprobs, topk_indices = lax.top_k(flat_logprobs, tokens_per_beam)
      expanded_parents = jnp.repeat(
          parent_tokens[:, None], tokens_per_beam, axis=1
      )
      next_state_candidates = dense_states[expanded_parents, topk_indices]

      limit = tokens_per_beam
      candidates_logprobs, candidates_indices, candidates_states = (
          topk_logprobs,
          topk_indices,
          next_state_candidates,
      )
    else:
      # --- SPARSE CSR LOOKUP ---
      # Transition to CSR logic once the state space becomes too sparse for dense tables.
      limit = max_branch_factors[step + 1]
      candidates_logprobs, candidates_indices, candidates_states = (
          generate_and_apply_logprobs_mask(
              flat_logprobs,
              flat_states,
              packed_csr,
              csr_indptr,
              limit,
              vocab_size,
          )
      )

    # --- SCORE & BEAM UPDATE ---
    # Update cumulative log-probability scores.
    scores = current_token_scores[:, :, None] + candidates_logprobs.reshape(
        batch_size, beam_size, limit
    )
    flat_scores = scores.reshape((batch_size, beam_size * limit))

    # Select the top beams for the next step.
    top_scores, flat_top_indices = lax.top_k(flat_scores, beam_size)

    # Recover token IDs and state transitions for the selected beams.
    flat_candidates_indices = candidates_indices.reshape(
        batch_size, beam_size * limit
    )
    flat_candidates_states = candidates_states.reshape(
        batch_size, beam_size * limit
    )

    top_tokens = jnp.take_along_axis(
        flat_candidates_indices, flat_top_indices, axis=1
    )
    current_transition_states = jnp.take_along_axis(
        flat_candidates_states, flat_top_indices, axis=1
    )

    # Update history and scores using one-hot contraction.
    top_beam_indices = flat_top_indices // limit
    token_buffer = _gather_beams(token_buffer, top_beam_indices)
    token_buffer = token_buffer.at[:, :, step + 1].set(top_tokens)

    current_token_scores = top_scores

  return token_buffer
