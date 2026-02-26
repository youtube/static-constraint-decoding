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

import jax
import jax.numpy as jnp
import numpy as np
from static_decoding.csr_utils import build_static_index
from static_decoding.decoding_jax import RandomModel
from static_decoding.decoding_jax import sparse_transition_jax


def run_validity_check(
    vocab_size, num_constraints, sid_len, beam_size, d_dense, key
):
  """Executes a single end-to-end validity test for a given configuration.

  This function synthesizes a random constraint graph, builds the STATIC index,
  and executes the constrained beam search using a dummy RandomModel. It then
  verifies that 100% of the decoded sequences are present in the original
  constraint set.

  Args:
    vocab_size (int): The total size of the token vocabulary.
    num_constraints (int): The number of valid Semantic IDs to generate for
      the test set.
    sid_len (int): The fixed length (L) of the Semantic IDs.
    beam_size (int): The number of beams to maintain during decoding.
    d_dense (int): The number of initial layers to handle with dense lookup
      tables before switching to the CSR matrix.
    key (jax.random.PRNGKey): The initial JAX PRNG key for random number
      generation.

  Returns:
    jax.random.PRNGKey: The updated JAX PRNG key for reproducibility.

  Raises:
    AssertionError: If any decoded sequence is not found in the original
      set of valid Semantic IDs.
  """
  print(
      f"  [Trial] V={vocab_size}, N={num_constraints}, L={sid_len},"
      f" beam={beam_size}, d={d_dense}"
  )

  # 1. Generate unique random Semantic IDs
  # SIDs are generated as (N, L) arrays
  sids = np.random.randint(
      0, vocab_size, size=(num_constraints, sid_len), dtype=np.int32
  )
  sids = np.unique(sids, axis=0)
  actual_n = sids.shape[0]

  # SIDs must be sorted for the CSR builder logic
  sids = sids[np.lexsort([sids[:, i] for i in range(sid_len - 1, -1, -1)])]

  # 2. Build the STATIC Index
  # This synthesizes the Start Mask, Dense Specialization tables, and CSR matrix
  p_csr, indptr, layer_max_branches, s_mask, d_mask, d_states = (
      build_static_index(sids, vocab_size, dense_lookup_layers=d_dense)
  )

  # 3. Move components to the accelerator device (JAX Arrays)
  p_csr_j = jnp.array(p_csr)
  indptr_j = jnp.array(indptr)
  s_mask_j = jnp.array(s_mask)
  d_mask_j = jnp.array(d_mask)
  d_states_j = jnp.array(d_states)

  # Ensure beam size doesn't exceed total valid items to avoid 'filler' beams
  test_beam_size = min(beam_size, actual_n)

  # 4. Instantiate the Random Model
  model = RandomModel(vocab_size)

  # 5. Run Constrained Beam Search
  # We use batch_size=2 to test parallel execution
  key, subkey = jax.random.split(key)
  outputs = sparse_transition_jax(
      model=model,
      key=subkey,
      batch_size=2,
      beam_size=test_beam_size,
      tokens_per_beam=10,
      start_token=0,
      max_sample_len=sid_len,
      vocab_size=vocab_size,
      max_branch_factors=layer_max_branches,
      packed_csr=p_csr_j,
      csr_indptr=indptr_j,
      start_mask=s_mask_j,
      dense_mask=d_mask_j,
      dense_states=d_states_j,
      d_dense=d_dense,
  )

  # 6. Verify results against the original set
  valid_set = {tuple(row) for row in sids}
  decoded_array = np.array(outputs)

  for b in range(2):
    for m in range(test_beam_size):
      decoded_sid = tuple(decoded_array[b, m])
      # Every sequence produced by the constrained beam search MUST be in the vocabulary
      assert decoded_sid in valid_set, (
          f"ERROR: Decoded invalid SID {decoded_sid} for d={d_dense}. "
          "This sequence was not in the original constraint set."
      )
  return key


def test_suite_end_to_end():
  """Runs a suite of validity checks across different trie depths and dense configurations."""
  # Initialize JAX PRNG key
  key = jax.random.PRNGKey(42)

  print(">>> Starting STATIC JAX Decoding Validity Test Suite...")

  # Define test configurations: (vocab, N, L, beam, d)
  configs = [
      (100, 50, 5, 10, 2),  # Standard d=2 config
      (50, 20, 8, 5, 1),  # Single-layer dense config
  ]

  for v, n, l, b, d in configs:
    try:
      key = run_validity_check(v, n, l, b, d, key)
      print(" [OK] Validity verified for configuration.")
    except Exception as e:
      print(f" [FAIL] Test failed for config d={d}: {e}")
      raise e

  print("\n>>> ALL DECODING VALIDITY TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
  test_suite_end_to_end()
