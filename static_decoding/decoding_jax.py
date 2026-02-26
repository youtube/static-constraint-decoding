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

from collections.abc import Callable
import functools
import jax
from jax import lax
import jax.numpy as jnp


def _gather_beams(x: jnp.ndarray, beam_indices: jnp.ndarray) -> jnp.ndarray:
  """Efficiently gathers beam data across a batch during the selection step.

  Uses one-hot contraction for TPU efficiency to select the top-M sequences
  from the candidate pool while preserving batch and history dimensions.

  Args:
      x: The source tensor to gather from.
          Shape: (batch_size, old_beam_size, ...).
      beam_indices: The indices of the beams to select.
          Shape: (batch_size, new_beam_size).

  Returns:
      The gathered tensor.
          Shape: (batch_size, new_beam_size, ...).
  """
  _, old_beam_size = x.shape[:2]

  # Create one-hot mask for the selection
  oh_beam_indices = jax.nn.one_hot(beam_indices, old_beam_size, dtype=jnp.int32)
  # Perform one-hot contraction via einsum
  return jnp.einsum("beo,bo...->be...", oh_beam_indices, x).astype(x.dtype)


def generate_and_apply_logprobs_mask(
    flat_logprobs: jnp.ndarray,
    flat_states: jnp.ndarray,
    packed_csr: jnp.ndarray,
    csr_indptr: jnp.ndarray,
    limit: int,
    vocab_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
  starts = csr_indptr[flat_states]
  actual_lens = csr_indptr[flat_states + 1] - starts

  # Create a grid of offsets to gather exactly 'limit' (K) candidates per state.
  offsets = jnp.arange(limit)
  # Broadcast: (B*M, 1) + (1, K) -> (B*M, K)
  gather_indices = starts[:, None] + offsets[None, :]

  # 2. Gather Data from CSR
  gathered_vals = jnp.take(
      packed_csr, gather_indices, axis=0, mode="fill", fill_value=0
  )
  candidate_token_ids = gathered_vals[:, :, 0]
  candidate_next_states = gathered_vals[:, :, 1]

  # 3. Logprob Gathering
  # Clamp indices to ensure we never gather from 'vocab_size' (which is OOB).
  # We use (vocab_size - 1) as a safe dummy index.
  safe_token_ids = jnp.clip(candidate_token_ids, max=vocab_size - 1)
  candidate_logprobs = jnp.take_along_axis(
      flat_logprobs, safe_token_ids, axis=1
  )

  # 4. Validity Masking
  valid_mask = offsets[None, :] < actual_lens[:, None]
  safe_logprobs = jnp.where(valid_mask, candidate_logprobs, -jnp.inf)

  return safe_logprobs, candidate_token_ids, candidate_next_states


class RandomModel:
  """A dummy model that acts like a Transformer but outputs random logits.

  Used to benchmark the throughput of the decoding harness without the
  computational overhead of a real neural network.
  """

  def __init__(self, vocab_size: int):
    self.vocab_size = vocab_size

  def __call__(
      self, input_ids: jnp.ndarray, key: jax.random.PRNGKey
  ) -> tuple[jnp.ndarray, jax.random.PRNGKey]:
    """Generates random logits for the next token prediction.

    Args:
        input_ids: Shape (batch_size, seq_len).
        key: JAX PRNG key.

    Returns:
        A tuple of (logits, next_key):
            - logits: Shape (batch_size, 1, vocab_size).
            - next_key: The updated PRNG key.
    """
    batch_size = input_ids.shape[0]
    key, subkey = jax.random.split(key)
    # Output random logits [0, 1]
    logits = jax.random.uniform(
        subkey, (batch_size, 1, self.vocab_size), minval=0, maxval=1
    )
    return logits, key


@functools.partial(
    jax.jit,
    static_argnames=(
        "model",
        "batch_size",
        "beam_size",
        "tokens_per_beam",
        "start_token",
        "max_sample_len",
        "vocab_size",
        "max_branch_factors",
        "d_dense",
    ),
)
def sparse_transition_jax(
    model: Callable,
    key: jax.random.PRNGKey,
    batch_size: int,
    beam_size: int,
    tokens_per_beam: int,
    start_token: int,
    max_sample_len: int,
    vocab_size: int,
    max_branch_factors: tuple[int, ...],
    packed_csr: jnp.ndarray,
    csr_indptr: jnp.ndarray,
    start_mask: jnp.ndarray,
    dense_mask: jnp.ndarray,
    dense_states: jnp.ndarray,
    d_dense: int = 2,
) -> jnp.ndarray:
  """Main harness for STATIC constrained beam search using a JAX model.

  Executes the full autoregressive decoding loop. Uses a hybrid approach,
  specializing the first `d_dense` codewords into dense lookup tables to
  maximize throughput on "hot" paths, and using a CSR matrix for the
  high-cardinality "sparse tail".

  Args:
      model: A callable that takes (input_ids, key) and returns
          (logits, new_key).
      key: Initial JAX PRNG key.
      batch_size: Number of sequences to decode in parallel (B).
      beam_size: Number of beams to maintain per sequence (M).
      tokens_per_beam: Number of candidate tokens to consider per beam.
      start_token: Token ID used to initiate decoding (BOS/PAD).
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
      dense_mask: d_dense-dimensional dense validity mask.
          Shape: (V,) * d_dense.
      dense_states: d_dense-dimensional dense state table.
          Shape: (V,) * d_dense.
      d_dense: Number of initial dense layers.
          NOTE: In practice, we only support d_dense=1 and d_dense=2
          (recommended).

  Returns:
      The decoded token sequences.
          Shape: (batch_size, beam_size, max_sample_len).
  """
  # --- 1. INITIAL STEP (Codeword 1) ---
  # Create start tokens to prime the model
  initial_input = jnp.full((batch_size, 1), start_token, dtype=jnp.int32)

  # Get logits from the model
  initial_logits, key = model(initial_input, key)
  raw_logprobs = jax.nn.log_softmax(initial_logits[:, 0, :])

  # Apply the root mask
  initial_logprobs = jnp.where(start_mask, raw_logprobs, -jnp.inf)
  top_logprobs, top_tokens = lax.top_k(initial_logprobs, beam_size)

  # Initialize decoding buffers.
  token_buffer = jnp.full(
      (batch_size, beam_size, max_sample_len),
      start_token,
      dtype=top_tokens.dtype,
  )
  token_buffer = token_buffer.at[:, :, 0].set(top_tokens)

  # Map Level-0 tokens to their initial trie state IDs (Token T -> ID T+1)
  current_transition_states = top_tokens + 1
  current_token_scores = top_logprobs

  # --- 2. AUTOREGRESSIVE LOOP (Codewords 2 to L) ---
  for step in range(max_sample_len - 1):
    # Prepare input: Flatten top tokens from previous step
    # Shape: (batch_size * beam_size, 1)
    flat_input_ids = top_tokens.reshape(batch_size * beam_size, 1)

    # Generate next-token logits from the model
    flat_logits, key = model(flat_input_ids, key)
    flat_logprobs = jax.nn.log_softmax(flat_logits[:, 0, :])
    flat_states = current_transition_states.reshape(batch_size * beam_size)

    # Apply hybrid dense/sparse masking
    if step < d_dense - 1:
      # --- DENSE SPECIALIZATION ---
      # Reconstruct previous token from state ID (Valid for d_dense=2)
      parent_tokens = (flat_states - 1).astype(jnp.int32)
      masks = dense_mask[parent_tokens]

      flat_logprobs = jnp.where(masks, flat_logprobs, -jnp.inf)

      topk_logprobs, topk_indices = lax.top_k(flat_logprobs, tokens_per_beam)

      # Map winners to next trie states using dense table
      next_state_candidates = dense_states[parent_tokens[:, None], topk_indices]

      limit = tokens_per_beam
      candidates_logprobs, candidates_indices, candidates_states = (
          topk_logprobs,
          topk_indices,
          next_state_candidates,
      )
    else:
      # --- SPARSE CSR LOOKUP ---
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
    scores = current_token_scores[:, :, None] + candidates_logprobs.reshape(
        batch_size, beam_size, limit
    )
    flat_scores = scores.reshape((batch_size, beam_size * limit))

    # Select the top beams for the next step
    top_scores, flat_top_indices = lax.top_k(flat_scores, beam_size)

    # Recover token IDs and state transitions
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

    # Update history and scores using one-hot contraction
    top_beam_indices = flat_top_indices // limit
    token_buffer = _gather_beams(token_buffer, top_beam_indices)
    token_buffer = token_buffer.at[:, :, step + 1].set(top_tokens)

    current_token_scores = top_scores

  return token_buffer
