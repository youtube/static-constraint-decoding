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

import numpy as np
from static_decoding.csr_utils import build_static_index


def verify_sparse_path(sid, d_val, L, d_states, p_csr, indptr):
  """Verifies that a single Semantic ID can be correctly traversed in the hybrid index.

  This function performs a manual walk of the trie structure to confirm that the
  hybrid dense/sparse representation maintains the connectivity of the original
  sequence. It validates the handover from the dense lookup table to the sparse
  CSR matrix and ensures the path leads to the correct terminal state.

  Args:
    sid (np.ndarray): The Semantic ID (token sequence) to verify.
    d_val (int): The number of initial layers handled by the dense lookup table.
    L (int): The total length of the Semantic ID.
    d_states (np.ndarray): The d-dimensional dense state table used for the
      initial `d_val` steps.
    p_csr (np.ndarray): The flattened Compressed Sparse Row (CSR) matrix
      containing the requisite trie transitions (cols and vals).
    indptr (np.ndarray): The CSR row pointer array.

  Raises:
    AssertionError: If a transition is missing in the CSR matrix or if the
      final state is not the terminal state (0).
  """
  # 1. Entry Point: Get the State ID reached after d tokens from the dense table
  prefix = tuple(sid[:d_val])
  current_state = d_states[prefix]

  # 2. Sparse Traversal: Follow CSR edges for the remaining codewords (d to L-1)
  for i in range(d_val, L):
    target_token = sid[i]

    # Identify the segment in CSR for the current state ID
    start_idx = indptr[current_state]
    end_idx = indptr[current_state + 1]
    edges = p_csr[start_idx:end_idx]

    # Search for the target token transition
    found = False
    for token, next_state in edges:
      if token == target_token:
        current_state = next_state
        found = True
        break

    assert found, (
        f"Transition for token {target_token} not found in state"
        f" {current_state} (Codeword index {i})"
    )

  # 3. Terminality: Verify the path ends in terminal state 0
  assert current_state == 0, (
      f"Path for SID {sid} did not end in terminal state 0 (ended in"
      f" {current_state})"
  )


def test_csr_configuration(sids, vocab_size, L, d_val):
  """Validates the structural integrity of a STATIC index for a specific configuration.

  This test builder constructs a STATIC index using the specified dense depth (`d_val`)
  and performs a suite of assertions to ensure correctness. It verifies:
  1.  Tensor Shapes: That dense masks/tables match the expected dimensions.
  2.  Start Mask: That all valid Level-0 tokens are marked.
  3.  Path Connectivity: That every SID in the dataset can be fully traversed.
  4.  Metadata: That branching factors (`layer_max_branches`) are correctly calculated.

  Args:
    sids (np.ndarray): A sorted array of Semantic IDs (shape [N, L]) representing
      the constraint vocabulary.
    vocab_size (int): The size of the token vocabulary.
    L (int): The length of each Semantic ID.
    d_val (int): The specific dense depth parameter (`d`) to verify.

  Raises:
    AssertionError: If any structural invariant or path verification fails.
  """
  print(f"  [Trial] Testing d={d_val}...")
  p_csr, indptr, lmb, s_mask, d_mask, d_states = build_static_index(
      sids, vocab_size, d=d_val
  )

  # A. Shape Verification
  expected_shape = (vocab_size,) * d_val
  assert d_mask.shape == expected_shape, f"d_mask shape mismatch for d={d_val}"
  assert (
      d_states.shape == expected_shape
  ), f"d_states shape mismatch for d={d_val}"

  # B. Start Mask Verification
  # Every token that starts a valid SID must be 'True' in the 1D start_mask
  valid_starts = np.unique(sids[:, 0])
  for token in valid_starts:
    assert (
        s_mask[token] == True
    ), f"Start mask missing valid Level-0 token: {token}"

  # C. Full Path Verification
  # For every SID, verify the handover from Dense lookup to CSR transitions
  for row in sids:
    verify_sparse_path(row, d_val, L, d_states, p_csr, indptr)

  # D. Metadata Verification
  # layer_max_branches must match SID length L for static shape compilation
  assert (
      len(lmb) == L
  ), f"layer_max_branches length {len(lmb)} does not match SID length {L}"

  # Correctness of branch indexing
  # lmb[0] is the root (number of valid starting tokens)
  assert lmb[0] == len(valid_starts), "Root branching factor mismatch"


def run_comprehensive_builder_test():
  """Main test harness for the STATIC index builder."""
  print(">>> Starting STATIC CSR Builder Validation...")

  # SETUP: Small vocabulary SID set for predictable testing
  # SIDs of length L=5 with overlaps to test trie branching
  vocab_size = 10
  L = 5
  sids = np.array(
      [
          [1, 2, 3, 4, 5],  # Base path
          [1, 2, 3, 6, 7],  # Branch at index 3
          [2, 3, 4, 5, 6],  # Separate root path
          [1, 2, 8, 9, 0],  # Branch at index 2
      ],
      dtype=np.int32,
  )

  # Sort SIDs as required by the builder logic
  sids = sids[np.lexsort([sids[:, i] for i in range(L - 1, -1, -1)])]

  # Test various dense lookup depths (d)
  # d can theoretically be anything from 1 to L-1
  for d in [1, 2, 3]:
    try:
      test_csr_configuration(sids, vocab_size, L, d)
      print(f"  [OK] Configuration d={d} passed.")
    except Exception as e:
      print(f"  [FAIL] Validation failed at d={d}: {e}")
      raise e

  print("\n>>> ALL CSR BUILDER TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
  run_comprehensive_builder_test()
