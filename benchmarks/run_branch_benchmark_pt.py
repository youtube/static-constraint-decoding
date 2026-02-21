# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

import gc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats
from static_decoding.csr_utils import build_static_index
from static_decoding.decoding_pt import generate_and_apply_logprobs_mask
import torch
from torch.profiler import profile
from torch.profiler import ProfilerActivity


def run_real_csr_benchmark_gpu(num_sequences=1_000_000, batch_beam=2, l_sid=8):
  """Hardware profiling benchmark for the STATIC masking kernel.

  For each factor B, sets vocab_size=B, synthesizes a new 1M-item CSR matrix,
  and measures the vectorized gather execution time via PyTorch Profiler.
  """
  # 1. SETUP: Factors matching the paper's scaling analysis
  FACTORS = [2**i for i in range(1, 19)]  # B=2 to 262,144
  ITERATIONS_PER_TRIAL = 1000  # Use 1000 for profiler stability

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  is_cuda = device.type == "cuda"

  # Set profiler activities conditionally based on hardware
  activities = [ProfilerActivity.CPU]
  if is_cuda:
    activities.append(ProfilerActivity.CUDA)

  print(f"Benchmarking on: {device} (Metric: Compiled Graph Time)")

  # GLOBAL WARMUP
  # Initializes CUPTI/Host and GPU state
  dummy = torch.randn(10, 10, device=device)
  with profile(activities=activities) as _:
    torch.mm(dummy, dummy)
    if is_cuda:
      torch.cuda.synchronize()

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
        sids_np, vocab_size, d=1
    )

    # Move Index to Device
    packed_csr = torch.tensor(packed_csr_np, dtype=torch.int32, device=device)
    csr_indptr = torch.tensor(indptr_np, dtype=torch.long, device=device)

    # Prepare dummy inputs
    num_states = len(indptr_np) - 1
    flat_states = torch.randint(
        0, num_states, (batch_beam,), dtype=torch.long, device=device
    )
    flat_logprobs = torch.randn(batch_beam, vocab_size, device=device)

    compiled_fn = None
    try:
      # 3. COMPILE SPECIALIZED KERNEL
      # Specialized for the constant B to enable Inductor optimizations
      def target_func(probs, states, csr, indptr):
        return generate_and_apply_logprobs_mask(
            probs,
            states,
            csr,
            indptr,
            limit=B,
            vocab_size=vocab_size,
            device=device,
        )

      # Captured using CUDA Graphs to minimize CPU launch overhead (on GPU)
      compiled_fn = torch.compile(target_func, mode="reduce-overhead")

      # 4. WARMUP & GRAPH CAPTURE
      for _ in range(10):
        compiled_fn(flat_logprobs, flat_states, packed_csr, csr_indptr)
      if is_cuda:
        torch.cuda.synchronize()

      # 5. PROFILING (1000 iterations for stable statistics)
      with profile(
          activities=activities,
          record_shapes=False,
      ) as prof:
        for _ in range(ITERATIONS_PER_TRIAL):
          compiled_fn(flat_logprobs, flat_states, packed_csr, csr_indptr)
        if is_cuda:
          torch.cuda.synchronize()

      # 6. METRIC EXTRACTION
      # Extract "Pure Kernel Time" from CompiledFxGraph events
      process_times = []
      for evt in prof.events():
        if evt.key.startswith("## Call CompiledFxGraph"):
          if is_cuda:
            # Prioritize device_time to satisfy the new PyTorch API
            t = (
                evt.device_time
                if hasattr(evt, "device_time")
                else getattr(evt, "cuda_time", 0)
            )
          else:
            # Fallback to CPU time if benchmarking locally on a CloudTop CPU
            t = getattr(evt, "cpu_time", 0)
          process_times.append(t)

      if not process_times:
        avg_us, ci = 0, 0
      else:
        process_times = np.array(process_times)
        avg_us = np.mean(process_times)
        sem = stats.sem(process_times)
        ci = sem * stats.t.ppf((1 + 0.95) / 2.0, len(process_times) - 1)

      print(f"    -> {avg_us:.2f} us +/- {ci:.2f}")
      results.append({"B": B, "Mean_us": avg_us, "CI_us": ci})

    except Exception as e:
      print(f"    Failed: {e}")

    # 7. AGGRESSIVE CLEANUP
    try:
      del compiled_fn
    except NameError:
      pass
    del packed_csr, csr_indptr, flat_states, flat_logprobs
    del sids_np, packed_csr_np, indptr_np
    gc.collect()
    if is_cuda:
      torch.cuda.empty_cache()

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
      label="STATIC PyTorch Time",
  )
  plt.fill_between(
      df["B"],
      df["Mean_ms"] - df["CI_ms"],
      df["Mean_ms"] + df["CI_ms"],
      color=color,
      alpha=0.15,
  )  # Lightly shaded region for standard deviation

  # 2. O(B) Linear Reference (scaled to ms, plotted only for B >= 4096)
  scale = df["Mean_ms"].iloc[-1] / df["B"].iloc[-1]
  df_ob = df[df["B"] >= 4096]

  # FIX: Applied the raw 'r' prefix to the LaTeX string
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

  # 3. Truncate the empty Y-space below 0.0015 ms
  ax.set_ylim(bottom=0.0015)

  # 4. Force the X-axis to display exact scalar numbers
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  plt.xticks(df["B"], labels=df["B"].astype(int), rotation=45)

  # 5. Force the Y-axis to display explicit decimal numbers (e.g., 0.1, 0.01)
  ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))

  # FIX: Applied the raw 'r' prefix to the LaTeX string
  plt.xlabel(r"Vocab Size ($|\mathcal{V}|$) & Branch Factor ($B$)")
  plt.ylabel("Compute Time (ms/step)")

  # Grid and Legend formatting
  plt.grid(True, which="both", linestyle="--", alpha=0.5)
  plt.legend(loc="lower right", frameon=True)

  # Adjust layout to ensure the rotated labels aren't cut off at the bottom
  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  df_results = run_real_csr_benchmark_gpu()
  print(df_results)
  # plot_benchmark(df_results)
