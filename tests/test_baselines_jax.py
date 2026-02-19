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

from benchmarks.baselines_jax import build_hash_bitmap
from benchmarks.baselines_jax import build_trie
from benchmarks.baselines_jax import generic_beam_search_jax
from benchmarks.baselines_jax import make_hash_bitmap_fn
from benchmarks.baselines_jax import make_ppv_mask_fn
from benchmarks.baselines_jax import make_trie_mask_fn
import jax
import jax.numpy as jnp
import numpy as np
from static_decoding.decoding_jax import RandomModel


def run_baseline_validity_check(
    method_name,
    mask_fn,
    sids,
    vocab_size,
    sid_len,
    beam_size,
    key,
):
  """Executes an integration test for a specific baseline constrained decoding method.

  This function instantiates a dummy `RandomModel` and runs the generic beam
  search
  harness (`generic_beam_search_jax`) using the provided `mask_fn`. It then
  validates
  that every generated sequence exists within the original constraint set
  `sids`.

  Args:
    method_name (str): The name of the baseline method being tested (e.g., "CPU
      Trie"). Used for error reporting and conditional logic (e.g., skipping
      Hash Bitmap checks).
    mask_fn (Callable): The JAX-compatible masking function to evaluate. This
      function must accept (logprobs, history, step) and return masked logprobs.
    sids (np.ndarray): The ground-truth set of valid Semantic IDs used to
      generate the constraints. Used for final membership verification.
    vocab_size (int): The size of the token vocabulary.
    sid_len (int): The length of the Semantic IDs (L).
    beam_size (int): The number of beams to use in the search.
    key (jax.random.PRNGKey): The JAX PRNG key for random number generation.

  Returns:
    jax.random.PRNGKey: The updated PRNG key for reproducibility.

  Raises:
    AssertionError: If a decoded sequence is invalid and the method is not "Hash
    Bitmap".
  """
  print(f"  [Trial] Method: {method_name} | V={vocab_size}, beam={beam_size}")

  # 1. Instantiate the Model
  # We use the same RandomModel mock as the STATIC tests for consistency
  model = RandomModel(vocab_size)

  # 2. Run the generic constrained beam search harness
  key, subkey = jax.random.split(key)
  outputs = generic_beam_search_jax(
      model=model,
      key=subkey,
      mask_fn=mask_fn,
      batch_size=2,  # Test parallel batch execution
      beam_size=beam_size,
      tokens_per_beam=10,
      max_sample_len=sid_len,
      start_token=0,  # Updated from pad_token
  )

  # 3. Verify results against the original set
  valid_set = {tuple(row) for row in sids}
  decoded_array = np.array(outputs)

  for b in range(2):
    for m in range(beam_size):
      decoded_sid = tuple(decoded_array[b, m])

      # Exception: Hash Bitmaps allow false positives by design
      if method_name == "Hash Bitmap":
        continue

      # Every sequence produced MUST be in the vocabulary
      assert decoded_sid in valid_set, (
          f"ERROR: {method_name} produced invalid SID {decoded_sid}. "
          "This sequence was not in the original constraint set."
      )
  return key


def test_baselines_suite():
  """Runs validity tests across all implemented JAX baselines."""
  # 1. SETUP: Environment and Synthetic Data
  key = jax.random.PRNGKey(42)
  vocab_size, sid_len, num_constraints, beam_size = 100, 5, 50, 10

  print(">>> Starting JAX Baseline Validity Test Suite...")

  # Generate unique random Semantic IDs (SIDs)
  sids = np.random.randint(
      0, vocab_size, size=(num_constraints, sid_len), dtype=np.int32
  )
  sids = np.unique(sids, axis=0)
  # SIDs must be sorted for PPV binary search logic
  sorted_sids = sids[
      np.lexsort([sids[:, i] for i in range(sid_len - 1, -1, -1)])
  ]
  sorted_sids_jax = jnp.array(sorted_sids)

  # 2. TEST: CPU TRIE
  # The Trie mask function accesses the python dictionary directly via callback.
  trie_root = build_trie(sorted_sids)
  trie_mask_fn = make_trie_mask_fn(trie_root, vocab_size)
  key = run_baseline_validity_check(
      "CPU Trie",
      trie_mask_fn,
      sorted_sids,
      vocab_size,
      sid_len,
      beam_size,
      key,
  )
  print(" [OK] CPU Trie validity verified.")

  # 3. TEST: HASH BITMAP
  # The Hash mask function uses the bitmap array via closure.
  bitmap = build_hash_bitmap(sorted_sids)
  hash_mask_fn = make_hash_bitmap_fn(bitmap)
  key = run_baseline_validity_check(
      "Hash Bitmap",
      hash_mask_fn,
      sorted_sids,
      vocab_size,
      sid_len,
      beam_size,
      key,
  )
  print(" [OK] Hash Bitmap execution verified (Note: FP allowed).")

  # 4. TEST: PPV (POINT-WISE VALIDATION)
  # The PPV mask function uses the sorted SIDs array via closure.
  # FIX: Use 'vocab_size' as top_k to perform an EXACT validity check.
  ppv_mask_fn = make_ppv_mask_fn(sorted_sids_jax, top_k=vocab_size)
  _ = run_baseline_validity_check(
      "PPV (Exact)",
      ppv_mask_fn,
      sorted_sids,
      vocab_size,
      sid_len,
      beam_size,
      key,
  )
  print(" [OK] PPV (Exact) validity verified.")

  print("\n>>> ALL BASELINE VALIDITY TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
  test_baselines_suite()
