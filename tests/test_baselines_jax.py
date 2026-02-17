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
from benchmarks.baselines_jax import generic_beam_search_jit
from benchmarks.baselines_jax import make_hash_bitmap_fn
from benchmarks.baselines_jax import make_ppv_mask_fn
from benchmarks.baselines_jax import make_trie_mask_fn
import jax
import jax.numpy as jnp
import numpy as np


def run_baseline_validity_check(
    method_name, mask_fn, mask_data, sids, vocab_size, sid_len, beam_size, key
):
  """Executes a validity check for a specific baseline masking algorithm."""
  print(f"  [Trial] Method: {method_name} | V={vocab_size}, beam={beam_size}")

  # 1. Run the generic constrained beam search harness
  key, subkey = jax.random.split(key)
  outputs = generic_beam_search_jit(
      key=subkey,
      mask_fn=mask_fn,
      mask_data=mask_data,
      batch_size=2,  # Test parallel batch execution
      beam_size=beam_size,
      tokens_per_beam=10,
      max_sample_len=sid_len,
      vocab_size=vocab_size,
      pad_token=0,
  )

  # 2. Verify results against the original set
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
  trie_root = build_trie(sorted_sids)
  trie_mask_fn = make_trie_mask_fn(trie_root, 2, beam_size, vocab_size, sid_len)
  key = run_baseline_validity_check(
      "CPU Trie",
      trie_mask_fn,
      jnp.array(0),
      sorted_sids,
      vocab_size,
      sid_len,
      beam_size,
      key,
  )
  print(" [OK] CPU Trie validity verified.")

  # 3. TEST: HASH BITMAP
  bitmap = build_hash_bitmap(sorted_sids)
  hash_mask_fn = make_hash_bitmap_fn()
  key = run_baseline_validity_check(
      "Hash Bitmap",
      hash_mask_fn,
      bitmap,
      sorted_sids,
      vocab_size,
      sid_len,
      beam_size,
      key,
  )
  print(" [OK] Hash Bitmap execution verified (Note: FP allowed).")

  # 4. TEST: PPV (POINT-WISE VALIDATION)
  # FIX: Use 'vocab_size' instead of '50' to perform an EXACT validity check.
  # The "Top-50" version is approximate and expected to fail on random sparse sets.
  ppv_mask_fn = make_ppv_mask_fn(top_k=vocab_size)
  _ = run_baseline_validity_check(
      "PPV (Exact)",
      ppv_mask_fn,
      sorted_sids_jax,
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
