import os
import sys

import numpy as np
import torch

# Ensure the package is importable from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from csr_utils import build_sparse_matrix_fast
from static_pt.decoding import sparse_transition_packed_torch


def run_validity_check(vocab_size, num_constraints, sid_len, beam_size, d_dense, device):
  """
  Executes a single end-to-end validity test for a given configuration.
  """
  print(f"  [Trial] V={vocab_size}, N={num_constraints}, L={sid_len}, beam={beam_size}, d={d_dense}")

  # 1. Generate unique random Semantic IDs
  # SIDs are generated as (N, L) arrays
  sids = np.random.randint(0, vocab_size, size=(num_constraints, sid_len), dtype=np.int32)
  sids = np.unique(sids, axis=0)
  actual_n = sids.shape[0]

  # SIDs must be sorted for the CSR builder logic
  sids = sids[np.lexsort([sids[:, i] for i in range(sid_len-1, -1, -1)])]

  # 2. Build the STATIC Index
  # This synthesizes the Start Mask, Dense Specialization tables, and CSR matrix
  p_csr, indptr, lmb, s_mask, d_mask, d_states = build_sparse_matrix_fast(
      sids, vocab_size, d=d_dense
  )

  # 3. Move components to the accelerator device
  p_csr_t = torch.tensor(p_csr, dtype=torch.long, device=device)
  indptr_t = torch.tensor(indptr, dtype=torch.long, device=device)
  s_mask_t = torch.tensor(s_mask, dtype=torch.bool, device=device)
  d_mask_t = torch.tensor(d_mask, dtype=torch.bool, device=device)
  d_states_t = torch.tensor(d_states, dtype=torch.long, device=device)

  # Ensure beam size doesn't exceed total valid items to avoid 'filler' beams
  test_beam_size = min(beam_size, actual_n)

  # 4. Run Constrained Beam Search
  # We use batch_size=2 to test parallel execution
  outputs = sparse_transition_packed_torch(
      batch_size=2,
      beam_size=test_beam_size,
      tokens_per_beam=10,
      pad_token=0,
      max_sample_len=sid_len,
      vocab_size=vocab_size,
      max_branch_factors=lmb,
      packed_csr=p_csr_t,
      csr_indptr=indptr_t,
      start_mask=s_mask_t,
      dense_mask=d_mask_t,
      dense_states=d_states_t,
      device=device,
      d_dense=d_dense
  )

  # 5. Verify results against the original set
  valid_set = {tuple(row) for row in sids}
  decoded_array = outputs.cpu().numpy()

  for b in range(2):
    for m in range(test_beam_size):
      decoded_sid = tuple(decoded_array[b, m])
      # Every sequence produced by the constrained beam search MUST be in the vocabulary
      assert decoded_sid in valid_set, (
          f"ERROR: Decoded invalid SID {decoded_sid} for d={d_dense}. "
          "This sequence was not in the original constraint set."
      )


def test_suite_end_to_end():
  """
  Runs a suite of validity checks across different trie depths and dense configurations.
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f">>> Starting STATIC Decoding Validity Test Suite on {device}...")

  # Define test configurations: (vocab, N, L, beam, d)
  configs = [
      (100, 50, 5, 10, 2),  # Standard d=2 config
      (50, 20, 8, 5, 1),   # Single-layer dense config
      #(200, 100, 4, 20, 3)  # Three-layer dense config (deeper specialization)
  ]

  for v, n, l, b, d in configs:
    try:
      run_validity_check(v, n, l, b, d, device)
      print(" [OK] Validity verified for configuration.")
    except Exception as e:
      print(f" [FAIL] Test failed for config d={d}: {e}")
      raise e

  print("\n>>> ALL DECODING VALIDITY TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
  test_suite_end_to_end()
