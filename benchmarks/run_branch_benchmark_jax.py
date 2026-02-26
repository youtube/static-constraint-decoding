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
import jax
from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats
from static_decoding.csr_utils import build_static_index
from static_decoding.decoding_jax import generate_and_apply_logprobs_mask


def run_real_csr_benchmark_oss(num_sequences=1_000_000, batch_beam=2, l_sid=8):
  """Open-source performance benchmark for the STATIC masking kernel.

  For each factor B, sets vocab_size=B, synthesizes a new 1M-item CSR matrix,
  and measures the vectorized gather execution time.
  """
  # 1. SETUP: Factors matching the paper's scaling analysis
  FACTORS = [2**i for i in range(1, 19)]  # B=2 to 262,144
  ITERATIONS_PER_TRIAL = 100
  NUM_TRIALS = 10  # Used to generate the 95% Confidence Interval

  device_type = jax.devices()[0].device_kind
  print(
      f"Benchmarking on: {device_type} (Metric: synchronized wall-clock time)"
  )

  results = []
  print(f">>> Recording Performance (Scaling B as vocab_size)...")

  for B in FACTORS:
    print(f"\n  [Trial] B={B} (vocab_size={B})")
    vocab_size = B

    # 2. GENERATE DATA (Real CSR synthesis PER factor B)
    print(f"    Generating Data (N={num_sequences}, V={vocab_size})...")
    sids_np = np.random.randint(
        0, vocab_size, size=(num_sequences, l_sid), dtype=np.int32
    )
    keys_np = [sids_np[:, i] for i in range(l_sid - 1, -1, -1)]
    sids_np = sids_np[np.lexsort(keys_np)]

    # Build STATIC Index (We only need the CSR components for the masking kernel)
    packed_csr_np, indptr_np, _, _, _, _ = build_static_index(
        sids_np, vocab_size, dense_lookup_layers=1
    )

    # Move Index to Accelerator Memory (HBM)
    packed_csr = jnp.array(packed_csr_np)
    csr_indptr = jnp.array(indptr_np)

    # Prepare dummy inputs based on the newly generated graph
    num_states = len(indptr_np) - 1
    flat_states = jax.random.randint(
        jax.random.PRNGKey(B), (batch_beam,), 0, num_states
    )
    flat_logprobs = jnp.zeros((batch_beam, vocab_size), dtype=jnp.float32)

    # Compile the specialized kernel for this constant B
    bench_jit = jit(
        generate_and_apply_logprobs_mask,
        static_argnames=("limit", "vocab_size"),
    )

    try:
      # 3. WARMUP
      # Triggers compilation and ensures the device is warmed up.
      _ = bench_jit(
          flat_logprobs, flat_states, packed_csr, csr_indptr, B, vocab_size
      )[0].block_until_ready()

      # 4. TIMED TRIALS
      trial_times = []
      res = None
      for _ in range(NUM_TRIALS):
        start = time.perf_counter()
        for _ in range(ITERATIONS_PER_TRIAL):
          res = bench_jit(
              flat_logprobs, flat_states, packed_csr, csr_indptr, B, vocab_size
          )

        # Force synchronization before taking the end timestamp
        res[0].block_until_ready()
        end = time.perf_counter()

        per_op_us = ((end - start) / ITERATIONS_PER_TRIAL) * 1e6
        trial_times.append(per_op_us)

      # 5. STATISTICAL EXTRACTION
      trial_times = np.array(trial_times)
      avg_us = np.mean(trial_times)
      sem = stats.sem(trial_times)
      ci = sem * stats.t.ppf((1 + 0.95) / 2.0, len(trial_times) - 1)

      print(f"    -> {avg_us:.2f} us +/- {ci:.2f}")
      results.append({"B": B, "Mean_us": avg_us, "CI_us": ci})

    except Exception as e:
      print(f"    Failed: {e}")

    # 6. AGGRESSIVE CLEANUP
    # Delete large arrays from host and device memory to avoid OOM on the next B
    del bench_jit, packed_csr, csr_indptr, flat_states, flat_logprobs
    del sids_np, packed_csr_np, indptr_np
    gc.collect()

  return pd.DataFrame(results)


def plot_benchmark(df):
  # Publication-specific formatting settings for KDD
  plt.rcParams.update({
      "font.size": 20,  # Base font size
      "axes.labelsize": 24,  # X and Y labels
      "axes.titlesize": 26,  # Title size
      "xtick.labelsize": 20,  # Tick labels
      "ytick.labelsize": 20,
      "legend.fontsize": 18,
      "pdf.fonttype": 42,  # Ensures standard font embedding for publications
      "ps.fonttype": 42,
  })

  # Professional aspect ratio for a two-column layout
  plt.figure(figsize=(14, 9))
  color = "orange"

  # 1. Convert microseconds to milliseconds
  df["Mean_ms"] = df["Mean_us"] / 1000.0
  df["CI_ms"] = df["CI_us"] / 1000.0

  # Plot Mean and CI with KDD styling
  plt.plot(
      df["B"],
      df["Mean_ms"],
      marker="o",
      markersize=12,
      linewidth=3,
      color=color,
      label="STATIC JAX/XLA Time",
  )
  plt.fill_between(
      df["B"],
      df["Mean_ms"] - df["CI_ms"],
      df["Mean_ms"] + df["CI_ms"],
      color=color,
      alpha=0.15,
  )  # Lightly shaded region for standard deviation

  # 2. O(B) Linear Reference (scaled to ms, plotted only for B >= 512)
  scale = df["Mean_ms"].iloc[-1] / df["B"].iloc[-1]
  df_ob = df[df["B"] >= 512]
  plt.plot(
      df_ob["B"],
      df_ob["B"] * scale,
      linestyle="--",
      linewidth=3,
      color="gray",
      alpha=0.5,
      label=r"$\mathcal{O}(B)$ Reference",
  )

  # Set logarithmic scales
  plt.xscale("log", base=2)
  plt.yscale("log", base=10)

  ax = plt.gca()

  # 3. Truncate the empty Y-space below 0.01 ms
  ax.set_ylim(bottom=0.01)

  # 4. Force the X-axis to display exact scalar numbers
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  plt.xticks(df["B"], labels=df["B"].astype(int), rotation=45)

  # 5. Force the Y-axis to display explicit decimal numbers (e.g., 0.1, 0.01)
  ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))

  # Update labels using LaTeX math mode
  plt.xlabel(r"Vocab Size ($|\mathcal{V}|$) & Branch Factor (B)")
  plt.ylabel("Compute Time (ms/step)")

  # Grid and Legend formatting
  plt.grid(True, which="both", linestyle="--", alpha=0.5)
  plt.legend(loc="lower right", frameon=True)

  # Adjust layout to ensure the rotated labels aren't cut off at the bottom
  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  df_results = run_real_csr_benchmark_oss()
  print(df_results)
  # plot_benchmark(df_results)
