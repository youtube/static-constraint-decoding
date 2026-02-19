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
import time
from benchmarks.baselines_jax import build_hash_bitmap
from benchmarks.baselines_jax import build_trie
from benchmarks.baselines_jax import generic_beam_search_jax
from benchmarks.baselines_jax import make_hash_bitmap_fn
from benchmarks.baselines_jax import make_ppv_mask_fn
from benchmarks.baselines_jax import make_trie_mask_fn
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from static_decoding.csr_utils import build_static_index
from static_decoding.decoding_jax import RandomModel
from static_decoding.decoding_jax import sparse_transition_jax

# =============================================================================
# 0. EXPERIMENT CONFIGURATION (Paper Appendix Table 4)
# =============================================================================
# Standard serving parameters used in the YouTube evaluation
BS, BM, TPB = 2, 70, 30
SID_LEN = 8
TRIALS = 5  # 100

METHODS = ["STATIC", "Trie", "Hash", "PPV-Approx", "PPV-Exact"]
BENCHMARKS = []

# Axis A: Scaling Vocabulary Size (|V|) with fixed |C|=10M
for v in [256, 512, 1024, 2048, 4096, 8192]:
  for method in METHODS:
    BENCHMARKS.append({"N": 10_000_000, "V": v, "Method": method})

# Axis B: Scaling Number of Constraints (|C|) with fixed |V|=2048
for n in [100_000, 1_000_000, 10_000_000, 20_000_000, 100_000_000]:
  for method in METHODS:
    # Optimization: Skip Trie at 100M due to host OOM risk
    if method == "Trie" and n >= 100_000_000:
      continue
    BENCHMARKS.append({"N": n, "V": 2048, "Method": method})

# Deduplicate and sort for sequential execution
BENCHMARKS = [dict(t) for t in {tuple(d.items()) for d in BENCHMARKS}]
BENCHMARKS.sort(key=lambda x: (x["N"], x["V"], x["Method"]))


def run_benchmarks():
  print(f"--- Starting Comparative Benchmark ({len(BENCHMARKS)} configs) ---")
  results = []
  cache = {}

  for run in BENCHMARKS:
    n, v, method = run["N"], run["V"], run["Method"]
    print(f"\n>>> Running: {method} | N={n}, V={v}")

    # --- 1. DATA GENERATION & CACHING ---
    cache_key = (n, v)
    if cache.get("key") != cache_key:
      print("  Generating Synthetic Constraints...")
      del cache
      gc.collect()
      sids_np = np.random.randint(0, v, size=(n, SID_LEN), dtype=np.int32)
      sids_np = sids_np[
          np.lexsort([sids_np[:, i] for i in range(SID_LEN - 1, -1, -1)])
      ]
      cache = {"key": cache_key, "sids_np": sids_np, "structs": {}}
    else:
      print("  Using Cached Data...")

    # Instantiate Model (Shared for both STATIC and Baselines)
    model = RandomModel(v)
    latencies = []

    try:
      # --- 2. EXECUTION: STATIC METHOD (Accelerator-Native) ---
      if method == "STATIC":
        if "static" not in cache["structs"]:
          # Build STATIC Index (d=2 dense specialization)
          p_csr, indptr, lmb, s_m, d_m, d_s = build_static_index(
              cache["sids_np"], v, d=2
          )
          cache["structs"]["static"] = (
              jnp.array(p_csr),
              jnp.array(indptr),
              lmb,
              jnp.array(s_m),
              jnp.array(d_m),
              jnp.array(d_s),
          )

        packed_csr, indptr, lmb, start_mask, dense_mask, dense_states = cache[
            "structs"
        ]["static"]

        # JIT Warmup
        _ = sparse_transition_jax(
            model,
            jax.random.key(0),
            BS,
            BM,
            TPB,
            0,  # start_token
            SID_LEN,
            v,
            lmb,
            packed_csr,
            indptr,
            start_mask,
            dense_mask,
            dense_states,
            d_dense=2,
        ).block_until_ready()

        for t in range(TRIALS):
          st = time.perf_counter()
          _ = sparse_transition_jax(
              model,
              jax.random.key(t),
              BS,
              BM,
              TPB,
              0,  # start_token
              SID_LEN,
              v,
              lmb,
              packed_csr,
              indptr,
              start_mask,
              dense_mask,
              dense_states,
              d_dense=2,
          ).block_until_ready()
          latencies.append((time.perf_counter() - st) * 1000)

      # --- 3. EXECUTION: BASELINE METHODS (Harness-based) ---
      else:
        mask_fn = None

        if method == "Trie":
          if "trie" not in cache["structs"]:
            cache["structs"]["trie"] = build_trie(cache["sids_np"])
          mask_fn = make_trie_mask_fn(cache["structs"]["trie"], v)

        elif method == "Hash":
          if "hash" not in cache["structs"]:
            cache["structs"]["hash"] = build_hash_bitmap(cache["sids_np"])
          mask_fn = make_hash_bitmap_fn(cache["structs"]["hash"])

        elif method in {"PPV-Approx", "PPV-Exact"}:
          k_val = 50 if method == "PPV-Approx" else v
          mask_fn = make_ppv_mask_fn(jnp.array(cache["sids_np"]), k_val)

        # Warmup
        _ = generic_beam_search_jax(
            model,
            jax.random.key(0),
            mask_fn,
            BS,
            BM,
            TPB,
            SID_LEN,
            start_token=0,
        ).block_until_ready()

        for t in range(TRIALS):
          st = time.perf_counter()
          _ = generic_beam_search_jax(
              model,
              jax.random.key(t),
              mask_fn,
              BS,
              BM,
              TPB,
              SID_LEN,
              start_token=0,
          ).block_until_ready()
          latencies.append((time.perf_counter() - st) * 1000)

      # --- 4. STATISTICAL LOGGING ---
      mean, std = np.mean(latencies), np.std(latencies)
      print(f"  [Result] {mean:.4f} ms ± {std:.4f}")
      results.append(
          {"Label": method, "N": n, "V": v, "Mean (ms)": mean, "Std (ms)": std}
      )

    except Exception as e:
      print(f"  [Error] Failed: {e}")
      results.append({
          "Label": method,
          "N": n,
          "V": v,
          "Mean (ms)": np.nan,
          "Std (ms)": np.nan,
      })

  # --- 5. REPORTING ---
  df_res = pd.DataFrame(results)
  print(
      "\n" + "=" * 50 + "\nREPRODUCTION RESULTS (APPENDIX TABLE 4)\n" + "=" * 50
  )
  print(df_res.to_string(index=False))
  # df_res.to_csv("table4_repro_results.tsv", sep='\t', index=False)


if __name__ == "__main__":
  run_benchmarks()
