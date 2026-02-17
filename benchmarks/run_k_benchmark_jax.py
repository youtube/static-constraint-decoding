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
from csr_utils import build_sparse_matrix_fast
import jax
from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from static.decoding_jax import generate_and_apply_logprobs_mask


def run_real_csr_benchmark_oss(
    vocab_size=2048, num_sequences=1_000_000, batch_beam=2, l_sid=8
):
  """Open-source performance benchmark for the STATIC masking kernel.

  Measures per-operation execution time using standard Python timing and
  accelerator synchronization (block_until_ready).
  """
  # 1. SETUP: Factors matching the paper's scaling analysis
  FACTORS = [2**i for i in range(1, 19)]  # K=2 to 262,144
  ITERATIONS_PER_TRIAL = 100
  NUM_TRIALS = 10  # Used to generate the 95% Confidence Interval

  device_type = jax.devices()[0].device_kind
  print(
      f"Benchmarking on: {device_type} (Metric: synchronized wall-clock time)"
  )

  # 2. GENERATE DATA (Real CSR synthesis)
  print(f">>> Generating Data (N={num_sequences}, V={vocab_size})...")
  sids_np = np.random.randint(
      0, vocab_size, size=(num_sequences, l_sid), dtype=np.int32
  )
  keys_np = [sids_np[:, i] for i in range(l_sid - 1, -1, -1)]
  sids_np = sids_np[np.lexsort(keys_np)]

  # Build STATIC Index
  packed_csr_np, indptr_np, lmb, _, _, _ = build_sparse_matrix_fast(
      sids_np, vocab_size
  )
  print(f"    Graph Built. Actual Max Branch Factor: {max(lmb)}")

  # Move Index to Accelerator Memory (HBM)
  packed_csr = jnp.array(packed_csr_np)
  csr_indptr = jnp.array(indptr_np)

  # Prepare dummy inputs
  num_states = len(indptr_np) - 1
  flat_states = jax.random.randint(
      jax.random.PRNGKey(0), (batch_beam,), 0, num_states
  )
  flat_logprobs = jnp.zeros((batch_beam, vocab_size), dtype=jnp.float32)

  results = []
  print(f">>> Recording Performance (Scaling K)...")

  for K in FACTORS:
    print(f"  [Trial] K={K}", end="")

    # Compile the specialized kernel for this constant K
    bench_jit = jit(
        generate_and_apply_logprobs_mask,
        static_argnames=("limit", "vocab_size"),
    )

    try:
      # 3. WARMUP
      # Triggers compilation and ensures the device is warmed up.
      # We block on the first result to ensure compilation completes.
      _ = bench_jit(
          flat_logprobs, flat_states, packed_csr, csr_indptr, K, vocab_size
      )[0].block_until_ready()

      # 4. TIMED TRIALS
      trial_times = []
      res = None
      for _ in range(NUM_TRIALS):
        # We time a block of iterations to minimize Python/Synchronization overhead.
        start = time.perf_counter()
        for _ in range(ITERATIONS_PER_TRIAL):
          # We only block at the end of the block to measure amortized hardware time.
          res = bench_jit(
              flat_logprobs, flat_states, packed_csr, csr_indptr, K, vocab_size
          )

        # Force synchronization before taking the end timestamp
        res[0].block_until_ready()
        end = time.perf_counter()

        # Calculate microseconds per operation
        per_op_us = ((end - start) / ITERATIONS_PER_TRIAL) * 1e6
        trial_times.append(per_op_us)

      # 5. STATISTICAL EXTRACTION
      trial_times = np.array(trial_times)
      avg_us = np.mean(trial_times)
      sem = stats.sem(trial_times)
      ci = sem * stats.t.ppf((1 + 0.95) / 2.0, len(trial_times) - 1)

      print(f" -> {avg_us:.2f} us +/- {ci:.2f}")
      results.append({"K": K, "Mean_us": avg_us, "CI_us": ci})

    except Exception as e:
      print(f" Failed: {e}")

    # Cleanup to avoid OOM across many jit specializations
    del bench_jit
    gc.collect()

  return pd.DataFrame(results)


def plot_benchmark(df):
  plt.figure(figsize=(14, 6))
  color = "orange"

  plt.plot(
      df["K"],
      df["Mean_us"],
      "o-",
      linewidth=2,
      color=color,
      label="Mean JAX/XLA Time (us)",
  )
  plt.fill_between(
      df["K"],
      df["Mean_us"] - df["CI_us"],
      df["Mean_us"] + df["CI_us"],
      color=color,
      alpha=0.2,
      label="95% Confidence Interval",
  )

  # O(K) Linear Reference
  scale = df["Mean_us"].iloc[-1] / df["K"].iloc[-1]
  plt.plot(
      df["K"],
      df["K"] * scale,
      "--",
      color="gray",
      alpha=0.5,
      label="O(K) Reference",
  )

  plt.xscale("log", base=2)
  plt.yscale("log", base=10)
  plt.xlabel("Branch Factor (K)", fontsize=12)
  plt.ylabel("Compute Time (us/op)", fontsize=12)
  plt.title("JAX Scaling Performance (Open Source Profiling)", fontsize=14)
  plt.grid(True, which="both", ls="-", alpha=0.3)
  plt.legend()
  plt.show()


if __name__ == "__main__":
  df_results = run_real_csr_benchmark_oss()
  print(df_results)
  # plot_benchmark(df_results)
