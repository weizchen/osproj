# Legacy Files Reference

This document explains the purpose of legacy/demo files that are kept for reference but are not part of the main research experiments.

## Legacy Files

### `benchmark_exp0.cu` / `bench0`
- **Purpose**: Original benchmark comparing standard CUDA launches vs persistent kernel
- **Status**: Superseded by `exp_a_pingpong.cu` and `exp_b_throughput.cu`
- **Why kept**: Useful reference for understanding evolution of benchmarks
- **Build**: `make bench0` or `make legacy`

### `exp0_microkernel.cu` / `exp0`
- **Purpose**: Original demonstration of persistent GPU microkernel
- **Status**: Superseded by unified GROSR runtime (`grosr_runtime.cu`)
- **Why kept**: Shows initial prototype implementation
- **Build**: `make exp0` or `make legacy`

### `exp1_syscalls.cu` / `exp1`
- **Purpose**: Proof-of-concept for GPU-initiated syscalls
- **Status**: Demonstrates GPU-CPU communication pattern, not part of main experiments
- **Why kept**: Useful reference for future work on GPU syscall mechanisms
- **Build**: `make exp1` or `make legacy`

### `run_experiments.py`
- **Purpose**: Python script for automating `benchmark_exp0.cu` execution and plotting
- **Status**: Only works with legacy `benchmark_exp0.cu`
- **Why kept**: May be useful as template for creating similar scripts for new experiments
- **Usage**: `python run_experiments.py` (requires `benchmark_exp0.cu`)

## Recommendation

For new development, use:
- **Experiments**: `exp_a_pingpong.cu`, `exp_b_throughput.cu`, `exp_c_graph_bfs.cu`
- **Runtime**: `grosr_runtime.h`, `grosr_*.cu` files
- **Tests**: `test_allocator.cu`

Legacy files are maintained for historical reference and educational purposes.

