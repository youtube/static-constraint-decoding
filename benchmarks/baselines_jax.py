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
from jax import jit
from jax import lax
import jax.numpy as jnp
import numpy as np


# =============================================================================
# 1. SHARED UTILITIES
# =============================================================================


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

  # Perform one-hot contraction via einsum for maximum XLA performance
  return jnp.einsum("beo,bo...->be...", oh_beam_indices, x).astype(x.dtype)


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


# =============================================================================
# 2. GENERIC BEAM SEARCH HARNESS
# =============================================================================


@functools.partial(
    jit,
    static_argnames=(
        "mask_fn",
        "batch_size",
        "beam_size",
        "tokens_per_beam",
        "max_sample_len",
        "vocab_size",
    ),
)
def generic_beam_search_jit(
    key,
    mask_fn,
    mask_data,
    batch_size,
    beam_size,
    tokens_per_beam,
    max_sample_len,
    vocab_size,
    pad_token=0,
):
  """A framework-agnostic harness for constrained beam search benchmarks.

  This function executes a standard autoregressive decoding loop but applies
  a provided `mask_fn` at every step, including the initial root step. This
  allows for a fair comparison between different constraint enforcement
  algorithms (Trie, PPV, Hash) using identical selection and history management.

  Args:
      key (jax.random.PRNGKey): PRNG key for mock logit generation.
      mask_fn (Callable): The baseline masking function to evaluate.
      mask_data (Any): The auxiliary data structure (Trie, Array, Bitmap)
        required by the mask_fn.
      batch_size (int): Number of parallel sequences.
      beam_size (int): Number of beams to maintain per sequence.
      tokens_per_beam (int): Number of candidate tokens to consider per beam.
      max_sample_len (int): The total decoding length (L).
      vocab_size (int): The size of the token vocabulary (V).
      pad_token (int): Token ID used for padding.

  Returns:
      jnp.ndarray: The decoded sequences of shape (batch_size, beam_size, L).
  """
  # --- 1. INITIAL STEP (Root) ---
  initial_logits, key = get_probs((batch_size, 1, vocab_size), key)
  initial_logprobs = jax.nn.log_softmax(initial_logits[:, 0, :])

  # Apply mask to the first token (step 0).
  # We use a dummy buffer for the prefix since the history is empty at start.
  dummy_buffer = jnp.zeros((batch_size, max_sample_len), dtype=jnp.int32)
  initial_logprobs = mask_fn(initial_logprobs, dummy_buffer, 0, mask_data)

  # Select initial beams
  top_logprobs, top_tokens = lax.top_k(initial_logprobs, beam_size)

  # Initialize decoding buffers
  token_buffer = jnp.full(
      (batch_size, beam_size, max_sample_len), pad_token, dtype=top_tokens.dtype
  )
  token_buffer = token_buffer.at[:, :, 0].set(top_tokens)
  cur_scores = top_logprobs

  # --- 2. AUTOREGRESSIVE LOOP (Steps 1 to L-1) ---
  for step in range(max_sample_len - 1):
    # Generate new logits from the model
    flat_logits, key = get_probs((batch_size * beam_size, 1, vocab_size), key)
    flat_logprobs = jax.nn.log_softmax(flat_logits[:, 0, :])

    # Apply the baseline constraint mask using the actual beam history
    # Note: We reshape the buffer to flatten (Batch * Beam) for the mask function
    flat_logprobs = mask_fn(
        flat_logprobs,
        token_buffer.reshape(-1, max_sample_len),
        step + 1,
        mask_data,
    )

    # Calculate cumulative scores and select top candidates
    topk_logprobs, topk_indices = lax.top_k(flat_logprobs, tokens_per_beam)
    scores = cur_scores[:, :, None] + jnp.reshape(
        topk_logprobs, (batch_size, beam_size, tokens_per_beam)
    )
    flat_scores = scores.reshape((batch_size, beam_size * tokens_per_beam))

    # Perform beam selection across the candidate pool
    top_scores, flat_top_indices = lax.top_k(flat_scores, beam_size)
    top_beam_indices = flat_top_indices // tokens_per_beam

    # Update state and history using optimized gather
    flat_candidate_tokens = topk_indices.reshape(
        batch_size, beam_size * tokens_per_beam
    )
    top_tokens = _gather_beams(flat_candidate_tokens, flat_top_indices)
    token_buffer = _gather_beams(token_buffer, top_beam_indices)

    # Commit selected tokens to the history buffer
    token_buffer = token_buffer.at[:, :, step + 1].set(top_tokens)
    cur_scores = top_scores

  return token_buffer


# =============================================================================
# 3. BASELINE: CPU TRIE (Dictionary-based)
# =============================================================================


def build_trie(sids):
  """Constructs a standard nested dictionary prefix tree on the CPU."""
  trie = {}
  for sid in sids:
    node = trie
    for t in sid:
      t_item = int(t)
      if t_item not in node:
        node[t_item] = {}
      node = node[t_item]
  return trie


def make_trie_mask_fn(trie_root, batch_size, beam_size, vocab_size, sid_len):
  """Creates a masking function that calls back to a CPU-based Trie.

  This baseline simulates the "pointer-chasing" behavior common in many
  production systems. It uses `jax.pure_callback` to pause accelerator
  execution and retrieve a validity mask from CPU memory at every step.
  """
  all_tokens_set = set(range(vocab_size))

  def python_callback(beams, step):
    """Internal Python logic to traverse the dictionary trie."""
    n = beams.shape[0]
    masks = np.zeros((n, vocab_size), dtype=bool)

    for i in range(n):
      node = trie_root
      valid = True
      # Traverse the trie to the current prefix depth
      for k in range(step):
        t = int(beams[i, k])
        if t in node:
          node = node[t]
        else:
          valid = False
          break

      if valid:
        valid_tokens = list(node.keys())
        # Optimization: use "Stop-List" logic for dense trie branches
        if len(valid_tokens) > vocab_size // 2:
          masks[i, :] = True
          masks[i, list(all_tokens_set.difference(valid_tokens))] = False
        else:
          masks[i, valid_tokens] = True
    return masks

  def mask_fn(logprobs, token_buffer, step, mask_data):
    """JAX-compatible wrapper for the trie callback."""
    # Determine dynamic batch dimension (handles step 0 vs step > 0)
    n = logprobs.shape[0]
    out_shape = jax.ShapeDtypeStruct((n, vocab_size), bool)

    # Perform the CPU callback using the correctly shaped logit mask
    mask = jax.pure_callback(
        functools.partial(python_callback, step=step), out_shape, token_buffer
    )
    return jnp.where(mask, logprobs, -1e9)

  return mask_fn


# =============================================================================
# 4. BASELINE: HASH BITMAP (Bloom Filter Style)
# =============================================================================


def build_hash_bitmap(sids):
  """Creates a static hash bitmap (Bloom filter style) for valid prefixes."""
  BITMAP_BITS = 30
  SIZE = 1 << BITMAP_BITS
  MULTIPLIER = 0x9E371

  bitmap = np.zeros(SIZE // 8, dtype=np.uint8)
  hashes = np.zeros(len(sids), dtype=np.int32)

  # Process each depth of the Semantic IDs to populate the bitmap
  for i in range(sids.shape[1]):
    tokens = sids[:, i]
    hashes = (hashes * MULTIPLIER + tokens) & (SIZE - 1)

    byte_idx = hashes // 8
    bit_idx = hashes % 8
    # Update the bitmap using bitwise OR
    np.bitwise_or.at(bitmap, byte_idx, (1 << bit_idx).astype(np.uint8))

  return jnp.array(bitmap)


def make_hash_bitmap_fn():
  """Creates a hash-based masking function optimized for JAX.

  This baseline is accelerator-native but suffers from potential false
  positives and lacks the O(1) candidate selection of the STATIC kernel.
  """
  BITMAP_BITS = 30
  SIZE = 1 << BITMAP_BITS
  MULTIPLIER = 0x9E371

  @jit
  def mask_fn(logprobs, token_buffer, step, mask_data):
    batch_beam, vocab_size = logprobs.shape

    def hash_prefix(seq):
      """Calculates the prefix hash for a single sequence."""
      h = jnp.int32(0)
      for i in range(token_buffer.shape[1]):
        h = jnp.where(i < step, (h * MULTIPLIER + seq[i]) & (SIZE - 1), h)
      return h

    # Calculate hashes for all beams in parallel
    prefix_hashes = jax.vmap(hash_prefix)(token_buffer)
    candidates = jnp.arange(vocab_size)

    # Calculate next-token hashes for all potential vocabulary extensions
    next_hashes = (
        prefix_hashes[:, None] * MULTIPLIER + candidates[None, :]
    ) & (SIZE - 1)

    # Probe the bitmap for each candidate
    byte_idx = next_hashes // 8
    bit_idx = next_hashes % 8
    is_set = (mask_data[byte_idx] & (1 << bit_idx)) != 0

    return jnp.where(is_set, logprobs, -1e9)

  return mask_fn


# =============================================================================
# 5. BASELINE: PPV (Point-wise Validation / Binary Search)
# =============================================================================


@functools.partial(jit, static_argnames=["M", "step"])
def ppv_batch_logic(flat_logprobs, history, step, sorted_sids, M):
  """Implements the PPV (Point-wise Validation) algorithm.

  PPV performs binary search across a sorted list of Semantic IDs to validate
  individual candidate extensions. This baseline corresponds to the method
  described in Ye et al. [30].
  """
  batch_size = flat_logprobs.shape[0]
  N = sorted_sids.shape[0]
  # Consider only the top-M most probable tokens from the model
  _, vt = lax.top_k(flat_logprobs, M)

  def get_batch_bounds():
    """Identifies the range in the sorted SID array matching the current prefix."""

    def search(search_type):
      low = jnp.zeros(batch_size, dtype=jnp.int32)
      high = jnp.full(batch_size, N, dtype=jnp.int32)

      def cond(state):
        return jnp.any(state[0] < state[1])

      def body(state):
        l, h = state
        mid = (l + h) // 2
        mid_seqs = sorted_sids[mid]

        # Identify the first discrepancy between the history and the midpoint
        not_eq = history != mid_seqs
        mask = jnp.arange(history.shape[1]) < step
        valid_diffs = not_eq & mask
        first_diff_idx = jnp.argmax(valid_diffs, axis=1)

        b_idx = jnp.arange(batch_size)
        val_target = history[b_idx, first_diff_idx]
        val_mid = mid_seqs[b_idx, first_diff_idx]

        # Decide which direction to move in the sorted list
        has_diff = valid_diffs[b_idx, first_diff_idx]
        move_right = jnp.where(
            has_diff, val_target > val_mid, (search_type == 1)
        )
        return jnp.where(move_right, mid + 1, l), jnp.where(move_right, h, mid)

      res_low, _ = lax.while_loop(cond, body, (low, high))
      return res_low

    return search(0), search(1)

  # Find the prefix bounds
  r_start, r_end = get_batch_bounds()
  low = jnp.repeat(r_start[:, None], M, axis=1)
  high = jnp.repeat(r_end[:, None], M, axis=1)

  def cond_p2(state):
    return jnp.any(state[0] < state[1])

  def body_p2(state):
    l, h = state
    mid = (l + h) // 2
    # Binary search for the specific next token candidate (vt)
    mid_tokens = sorted_sids[mid, step]
    greater = vt > mid_tokens
    return jnp.where(greater, mid + 1, l), jnp.where(greater, h, mid)

  # Execute binary search for next tokens
  low, _ = lax.while_loop(cond_p2, body_p2, (low, high))

  # Final exact match verification
  safe_low = jnp.minimum(low, N - 1)
  valid_range = (low >= r_start[:, None]) & (low < r_end[:, None])
  matched_token = sorted_sids[safe_low, step]
  final_valid = valid_range & (vt == matched_token)

  # Construct the dense logit mask
  result_mask = jnp.zeros((batch_size, flat_logprobs.shape[1]), dtype=bool)
  return result_mask.at[jnp.arange(batch_size)[:, None], vt].max(final_valid)


def make_ppv_mask_fn(top_k):
  """Creates a PPV-based masking function."""

  def mask_fn(flat_logprobs, token_buffer, step, mask_data):
    batch_size = flat_logprobs.shape[0]
    max_len = token_buffer.shape[1]

    # Execute the PPV binary search logic
    masks = ppv_batch_logic(
        flat_logprobs,
        token_buffer.reshape(batch_size, max_len),
        step,
        mask_data,
        top_k,
    )
    return jnp.where(masks, flat_logprobs, -1e9)

  return mask_fn
