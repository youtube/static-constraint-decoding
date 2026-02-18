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

import gc
import numpy as np


def build_static_index(fresh_sids, vocab_size=2048, d=2):
  """
  Constructs a STATIC index (Sparse Transition-Accelerated Trie Index) from Semantic IDs.

  This function transforms a prefix tree (trie) into a static, accelerator-compatible
  representation. It uses a hybrid approach:
  1. A d-dimensional dense lookup table for the "hot" initial layers (first d codewords).
  2. A Compressed Sparse Row (CSR) matrix for the high-cardinality "sparse tail"
      (all codewords from d+1 to L).

  Args:
      fresh_sids (np.ndarray): Sorted NumPy array of shape (N, L) containing the
          tokenized Semantic IDs for the entire corpus.
      vocab_size (int): The model's token vocabulary size.
      d (int): Number of initial layers to handle with dense lookups.
                d=2 corresponds to a V x V table.

  Returns:
      packed_csr (np.ndarray): A flat array of [token, next_state] pairs for all transitions;
                               contains the "cols" and "vals" of the CSR matrix.
      indptr (np.ndarray): The CSR row pointer array identifying segments in packed_csr.
      layer_max_branches (tuple): Maximum branching factor for each level L,
                                  required for static-shape compilation.
      start_mask (np.ndarray): 1D boolean mask for the very first token (Level 0).
      dense_mask (np.ndarray): d-dimensional boolean mask for the first d tokens.
      dense_states (np.ndarray): d-dimensional state ID table for the first d tokens.
  """
  # N is the number of items in the corpus; L is the length of each Semantic ID.
  N, L = fresh_sids.shape
  if d >= L:
    raise ValueError(f"d ({d}) must be less than the Semantic ID length L ({L}).")

  # --- 1. INITIAL LEVEL-0 MASK ---
  # We identify which tokens are valid starting points.
  # This is always a 1D vector of length 'vocab_size'.
  start_mask = np.zeros(vocab_size, dtype=bool)
  start_mask[np.unique(fresh_sids[:, 0])] = True

  # --- 2. VECTORIZED TRIE NODE IDENTIFICATION ---
  # To convert the trie to a static matrix, we first identify unique nodes (prefixes).
  # Since fresh_sids is sorted, a new node starts whenever a token at a specific
  # depth differs from the token at the same depth in the previous sequence.
  diff = (fresh_sids[1:] != fresh_sids[:-1])
  first_diff = np.full(N - 1, L, dtype=np.int8)
  has_diff = diff.any(axis=1)

  # 'first_diff' stores the index of the first token that changed between rows.
  first_diff[has_diff] = diff[has_diff].argmax(axis=1)

  # 'is_new' is a boolean matrix of shape (N, L) identifying unique prefixes.
  is_new = np.zeros((N, L), dtype=bool)
  is_new[0, :] = True  # The first item in the sorted list is always a "new" path.
  for depth in range(L):
    is_new[1:, depth] = (first_diff <= depth)

  # --- 3. STATE ID ASSIGNMENT ---
  # We map every unique prefix to a unique integer 'State ID'.
  # Level-0 nodes (tokens of length 1) are assigned IDs 1 to vocab_size.
  state_ids = np.zeros((N, L - 1), dtype=np.int32)
  state_ids[:, 0] = fresh_sids[:, 0].astype(np.int32) + 1

  depth_id_ranges = []
  current_offset = vocab_size + 1

  # Iterate through each depth to assign IDs to deeper nodes.
  for depth in range(1, L - 1):
    mask = is_new[:, depth]
    num_new = np.sum(mask)
    start_id = current_offset
    end_id = current_offset + num_new

    # Track the ID range for this depth to calculate branch factors later.
    depth_id_ranges.append((start_id, end_id))

    # Assign unique IDs and use 'maximum.accumulate' to fill the IDs for
    # duplicate prefixes in the sorted array.
    state_ids[mask, depth] = np.arange(start_id, end_id, dtype=np.int32)
    state_ids[:, depth] = np.maximum.accumulate(state_ids[:, depth])
    current_offset += num_new

  num_states = current_offset

  # --- 4. EDGE COLLECTION ---
  # We collect all parent -> child transitions.
  # An edge is defined as: (Parent State ID, Token) -> Child State ID.
  all_p, all_t, all_c = [], [], []
  for depth in range(1, L):
    mask = is_new[:, depth]
    p = state_ids[mask, depth-1]  # ID of the prefix of length 'depth'
    t = fresh_sids[mask, depth].astype(np.int32)  # The next token

    # If we are at the last token, the child state is 0 (terminal).
    c = state_ids[mask, depth] if depth < L - 1 else np.zeros_like(p, dtype=np.int32)

    all_p.append(p)
    all_t.append(t)
    all_c.append(c)

  # --- 5. DENSE SPECIALIZATION ---
  # For the first 'd' layers, we synthesize a dense multi-dimensional tensor.
  # This allows fast O(1) indexing during the initial (and most frequent) steps.
  dense_shape = tuple([vocab_size] * d)
  dense_mask = np.zeros(dense_shape, dtype=bool)
  dense_states = np.zeros(dense_shape, dtype=np.int32)

  # We index the table using the first 'd' codewords.
  indices = tuple(fresh_sids[:, i].astype(np.int32) for i in range(d))

  # The value in 'dense_states' is the ID of the node reached after 'd' tokens.
  final_dense_ids = state_ids[:, d - 1]

  dense_mask[indices] = True
  dense_states[indices] = final_dense_ids

  # --- 6. CSR CONSTRUCTION ---
  # Combine all collected edges into a flat format.
  parents = np.concatenate(all_p)
  tokens = np.concatenate(all_t)
  children = np.concatenate(all_c)

  del state_ids, is_new; gc.collect()

  # 'counts' helps build 'indptr', which points to the start of a state's edges.
  counts = np.bincount(parents, minlength=num_states)
  indptr = np.zeros(num_states + 1, dtype=np.int32)
  indptr[1:] = np.cumsum(counts)

  # --- 7. LAYER MAX BRANCHES (Static Compilation Metadata) ---
  # Accelerator compilers require static output shapes. We calculate the
  # maximum number of possible child tokens at each level of the trie.
  lmb = [np.sum(start_mask)]  # Max branches out of the root.

  # Max branches out of Level-0 tokens (IDs 1 to vocab_size).
  l0_counts = counts[1:vocab_size + 1]
  lmb.append(int(l0_counts.max()) if len(l0_counts) > 0 else 0)

  # Max branches out of subsequent nodes.
  for (start_id, end_id) in depth_id_ranges:
    if start_id < len(counts):
      layer_counts = counts[start_id:end_id]
      lmb.append(int(layer_counts.max()) if len(layer_counts) > 0 else 0)
    else:
      lmb.append(0)

  # Pad to SID length L.
  while len(lmb) < L:
    lmb.append(1)

  # --- 8. FINAL PACKING ---
  # We create a flat transition table [Token, NextState].
  # We add a padding row at the end to handle OOB indexing in compiled kernels.
  raw_indices = np.concatenate([tokens, np.full(vocab_size, vocab_size, dtype=np.int32)])
  raw_data = np.concatenate([children, np.zeros(vocab_size, dtype=np.int32)])
  indptr = np.append(indptr, indptr[-1] + vocab_size)

  # 'ascontiguousarray' ensures optimal GPU HBM burst throughput.
  packed_csr = np.ascontiguousarray(np.vstack([raw_indices, raw_data]).T)

  return packed_csr, indptr, tuple(lmb), start_mask, dense_mask, dense_states
