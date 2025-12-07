# Repository Cleanup Summary

## Files Removed

### Deleted Files
1. **`experiment0_persistent_runtime.cu`**
   - **Reason**: Duplicate implementation of `exp0_microkernel.cu`
   - **Status**: Not referenced in Makefile, redundant code
   - **Impact**: None - functionality preserved in `exp0_microkernel.cu`

## Files Reorganized

### Legacy Files (Kept for Reference)
The following files are kept but marked as legacy/reference-only:

1. **`benchmark_exp0.cu`** - Original benchmark (superseded by `exp_a_pingpong.cu` and `exp_b_throughput.cu`)
2. **`exp0_microkernel.cu`** - Original microkernel demo (superseded by unified `grosr_runtime.cu`)
3. **`exp1_syscalls.cu`** - GPU syscall proof-of-concept (not part of main experiments)
4. **`run_experiments.py`** - Python script for legacy `benchmark_exp0.cu` only

**Build**: Use `make legacy` to build legacy executables, or `make main` for current experiments only.

## Current Project Structure

### Core Runtime (Required)
- `grosr_runtime.h` - Unified GROSR runtime header
- `grosr_allocator.cu` - GPU-side slab allocator
- `grosr_queue.cu` - Task queue management
- `grosr_runtime.cu` - Persistent runtime kernel

### Main Experiments (Current)
- `exp_a_pingpong.cu` - Experiment A: Ping-Pong Latency
- `exp_b_throughput.cu` - Experiment B: Throughput Benchmark
- `exp_c_graph_bfs.cu` - Experiment C: Graph BFS Macrobenchmark
- `test_allocator.cu` - Allocator unit tests

### Documentation
- `README.md` - Project overview and usage
- `RESEARCH_PLAN.md` - Detailed research plan
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `LEGACY_FILES.md` - Explanation of legacy files
- `GEMINI_CHAT.md` - Historical reference (chat logs)
- `GPT_CHAT.md` - Historical reference (chat logs)

### Build System
- `Makefile` - Build system with `main`, `legacy`, and `all` targets

## Build Targets

- **`make main`** - Build main experiments (recommended)
- **`make all`** - Build everything (main + legacy)
- **`make legacy`** - Build only legacy demos
- **`make clean`** - Remove all build artifacts

## Impact

- **Reduced redundancy**: Removed duplicate `experiment0_persistent_runtime.cu`
- **Clearer structure**: Separated main experiments from legacy demos
- **Better organization**: Updated README with clear file categorization
- **Maintained compatibility**: All original functionality preserved

## Recommendations

1. Use `make main` for regular development
2. Legacy files are kept for reference but not actively maintained
3. New experiments should follow the pattern of `exp_a_*`, `exp_b_*`, `exp_c_*`
4. See `LEGACY_FILES.md` for details on legacy file purposes

