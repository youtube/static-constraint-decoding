# STATIC: An Accelerator-Native Framework for Constrained Decoding

This repository provides the official implementation of the **STATIC (Sparse Transition-Accelerated Trie Index for Constrained decoding)** framework. STATIC is a high-performance method for enforcing outputs to stay within a prespecified set during autoregressive decoding from large language models, designed for maximum efficiency on modern hardware accelerators like GPUs and TPUs.

This implementation includes:
- Core algorithms for both JAX/TPU and PyTorch/GPU settings.
- Comprehensive benchmarks against standard baselines (Naive Trie-based, Hashmap-based, and PPV).
- A full suite of unit tests for correctness and validity.

## Key Features

- **Accelerator-Native Design**: The core masking kernel is implemented as a single, vectorized operation, avoiding expensive CPU-accelerator synchronization and pointer-chasing common in traditional trie-based methods.
- **Hybrid Data Structure**: STATIC uses a novel hybrid index. It represents the "hot" initial layers of a prefix tree with a dense lookup table for O(1) access and the high-cardinality "sparse tail" with a Compressed Sparse Row (CSR) matrix for memory efficiency.
- **High Performance**: Achieves near-constant-time (O(1)) performance with respect to the total number of constraints, and logarithmic performance (O(log K)) relative to the branching factor (K), significantly outperforming traditional baselines.
- **Framework Agnostic**: Includes end-to-end, tested implementations for both major deep learning frameworks: JAX and PyTorch.

## How It Works

The core of STATIC is a two-part process: an offline indexing step and an online masking step.

1.  **Offline Indexing (`build_static_index`)**:
    - Takes a large set of valid token sequences (e.g., millions of Semantic IDs) as input.
    - Analyzes the prefix structure and converts the implicit trie into the hybrid dense/sparse representation.
    - It synthesizes several components:
        - A `start_mask` to validate the very first token.
        - A `dense_mask` and `dense_states` tensor to handle the first `d` tokens.
        - A `packed_csr` and `csr_indptr` matrix to represent all transitions beyond depth `d`.

2.  **Online Masking (`sparse_transition_jax`/`_torch`)**:
    - During each step of autoregressive decoding (e.g., beam search), the model's predicted log-probabilities are masked.
    - For the first `d` steps, valid next tokens are retrieved in O(1) from the dense tables.
    - For all subsequent steps, the `generate_and_apply_logprobs_mask` kernel performs a vectorized burst-read from the CSR matrix to fetch all valid continuations for all beams in parallel.
    - This provides the final mask, which is applied to the log-probabilities before selecting the next tokens.

This design ensures that the cost of masking is independent of the total number of constraints, making it highly scalable.

## Repository Structure

```
.
├── static_decoding/
│   ├── csr_utils.py                    # Core STATIC index construction logic (NumPy-based).
│   ├── decoding_jax.py                 # Core STATIC decoding loop for JAX.
│   └── decoding_pt.py                  # Core STATIC decoding loop for PyTorch.
│
├── benchmarks/
│   ├── baselines_jax.py                # JAX implementations of Trie, Hash bitmap, and PPV baselines.
│   ├── run_comparative_benchmark_jax.py  # Script to compare STATIC against baselines.
│   ├── run_k_benchmark_jax.py            # JAX kernel scaling benchmark (vs. branch factor K).
│   └── run_k_benchmark_pt.py             # PyTorch kernel scaling benchmark.
│
├── tests/
│   ├── test_baselines_jax.py           # Validity tests for baseline algorithms.
│   ├── test_csr_builder.py             # Unit tests for the STATIC index builder.
│   ├── test_jax_decoding.py            # End-to-end validity tests for the JAX decoder.
│   └── test_pt_decoding.py             # End-to-end validity tests for the PyTorch decoder.
│
└── README.md                           # This file.
```

## Getting Started

### Prerequisites

This project requires Python and the following core libraries:
- JAX
- PyTorch
- NumPy
- Pandas (for benchmarks)
- Matplotlib & SciPy (for plotting benchmark results)

You can install them via pip:
```bash
pip install -q numpy pandas matplotlib scipy
pip install -q torch --index-url https://download.pytorch.org/whl/cu118
pip install -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html # For TPU
# or
pip install -q jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # For GPU
```

### Installation

Install the package in editable mode by running the command:

```
pip install -e .
```

### Usage Example

The repository includes `example.ipynb`, a notebook that provides a simple, hands-on demonstration of the STATIC framework using the JAX implementation. The notebook walks through the core steps:

1.  Synthesizing Data: Generating a set of random, valid Semantic IDs (SIDs) to define the constraint vocabulary.
2.  Building the Index: Using `build_static_index` to convert the SIDs into the hybrid dense/sparse STATIC representation.
3.  Running Decoding: Executing a constrained beam search with `sparse_transition_jax` and a mock `RandomModel` to generate valid sequences on an accelerator.

To run the example, start a Jupyter Notebook server and open `example.ipynb`.

### Running Tests

The validity of the implementation can be verified by running the test suites. Each test script can be executed directly after installing the repository.

```bash
# Test the CSR index builder
python tests/test_csr_builder.py

# Test the end-to-end JAX decoding loop
python tests/test_jax_decoding.py

# Test the end-to-end PyTorch decoding loop
python tests/test_pt_decoding.py

# Test the JAX baseline implementations
python -m tests.test_baselines_jax
```

### Running Benchmarks

The repository includes scripts to reproduce the performance benchmarks.

```bash
# Run the comparative benchmark (STATIC vs. baselines)
python -m benchmarks.run_comparative_benchmark_jax

# Run the kernel-level scaling analysis for JAX
python benchmarks/run_k_benchmark_jax.py

# Run the kernel-level scaling analysis for PyTorch
python benchmarks/run_k_benchmark_pt.py
```
The benchmark scripts will print results directly to the console in a formatted table.

## License

This project is licensed under the Apache License, Version 2.0. See the license headers in the source files for more details.

## Notes

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

This project is intended for demonstration purposes only. It is not intended for use in a production environment.
