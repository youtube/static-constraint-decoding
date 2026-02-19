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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from static_decoding.csr_utils import build_static_index
from static_decoding.decoding_pt import generate_and_apply_logprobs_mask
import torch
from torch.profiler import profile
from torch.profiler import ProfilerActivity


def run_real_csr_benchmark_pytorch(
    vocab_size=2048, num_sequences=10_000_000, batch_beam=2, l_sid=8
):
  """Hardware profiling benchmark for the STATIC masking kernel.

  Measures execution time for the masking kernel specialized for constant K,
  supporting both CUDA (via CUDA Graphs) and CPU (via standard Inductor
  kernels).
  """
  # 1. SETUP
  FACTORS = [2**i for i in range(1, 19)]  # K=2 to 262,144
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Determine profiling settings based on hardware
  is_cuda = device.type == "cuda"
  activities = [ProfilerActivity.CPU]
  if is_cuda:
    activities.append(ProfilerActivity.CUDA)

  # Mode "reduce-overhead" is GPU-only (CUDA Graphs)
  compile_mode = "reduce-overhead" if is_cuda else "default"

  print(f"Benchmarking on: {device} (Mode: {compile_mode})")

  results = []

  # 2. GLOBAL WARMUP
  dummy = torch.randn(10, 10, device=device)
  with profile(activities=activities) as _:
    torch.mm(dummy, dummy)
    if is_cuda:
      torch.cuda.synchronize()

  # 3. GENERATE DATA
  print(f">>> Generating Data (N={num_sequences}, V={vocab_size})...")
  sids_np = np.random.randint(
      0, vocab_size, size=(num_sequences, l_sid), dtype=np.int32
  )
  keys_np = [sids_np[:, i] for i in range(l_sid - 1, -1, -1)]
  sids_np = sids_np[np.lexsort(keys_np)]

  packed_csr_np, indptr_np, lmb, _, _, _ = build_static_index(
      sids_np, vocab_size
  )
  print(f"    Graph Built. Actual Max Branch Factor: {max(lmb)}")

  packed_csr = torch.tensor(packed_csr_np, dtype=torch.int32, device=device)
  csr_indptr = torch.tensor(indptr_np, dtype=torch.long, device=device)

  num_states = len(indptr_np) - 1
  flat_states = torch.randint(
      0, num_states, (batch_beam,), dtype=torch.long, device=device
  )
  flat_logprobs = torch.randn(batch_beam, vocab_size, device=device)

  print(f">>> Recording Hardware Trace (Scaling K)...")

  for K in FACTORS:
    print(f"  [Trial] K={K}", end="")
    try:
      # 4. COMPILE SPECIALIZED KERNEL
      def target_func(probs, states, csr, indptr):
        return generate_and_apply_logprobs_mask(
            probs,
            states,
            csr,
            indptr,
            limit=K,
            vocab_size=vocab_size,
            device=device,
        )

      compiled_fn = torch.compile(target_func, mode=compile_mode)

      # 5. WARMUP
      for _ in range(10):
        compiled_fn(flat_logprobs, flat_states, packed_csr, csr_indptr)
      if is_cuda:
        torch.cuda.synchronize()

      # 6. PROFILING (1000 iterations)
      ITERATIONS = 1000
      with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(ITERATIONS):
          compiled_fn(flat_logprobs, flat_states, packed_csr, csr_indptr)
        if is_cuda:
          torch.cuda.synchronize()

      # 7. METRIC EXTRACTION
      process_times = []
      for evt in prof.events():
        if evt.key.startswith("## Call CompiledFxGraph"):
          # Extraction logic optimized for device type
          if is_cuda:
            t = (
                evt.device_time
                if hasattr(evt, "device_time")
                else getattr(evt, "cuda_time", 0)
            )
          else:
            t = evt.cpu_time_total  # Primary metric for CPU compiled kernels
          process_times.append(t)

      if not process_times:
        avg_us, ci = 0, 0
      else:
        process_times = np.array(process_times)
        avg_us = np.mean(process_times)
        sem = stats.sem(process_times)
        ci = sem * stats.t.ppf((1 + 0.95) / 2.0, len(process_times) - 1)

      print(f" -> {avg_us:.2f} us +/- {ci:.2f}")
      results.append({"K": K, "Mean_us": avg_us, "CI_us": ci})

      del compiled_fn
      gc.collect()

    except Exception as e:
      print(f" Failed: {e}")

  return pd.DataFrame(results)


def plot_benchmark(df):
  plt.figure(figsize=(14, 6))
  plt.plot(
      df["K"],
      df["Mean_us"],
      "o-",
      linewidth=2,
      color="orange",
      label="Mean Kernel Time (us)",
  )
  plt.fill_between(
      df["K"],
      df["Mean_us"] - df["CI_us"],
      df["Mean_us"] + df["CI_us"],
      color="orange",
      alpha=0.2,
  )

  # O(K) Reference
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
  plt.xlabel("Branch Factor (K)")
  plt.ylabel("Compute Time (us/op)")
  plt.title("STATIC Scaling Performance (Hardware Profiling)")
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.show()


if __name__ == "__main__":
  df_results = run_real_csr_benchmark_pytorch()
  print(df_results)
  # plot_benchmark(df_results)
