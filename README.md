# GROSR: GPU-Resident Operating System Runtime

**GROSR** eliminates the CPU control-plane bottleneck for fine-grained, dynamic GPU workloads by moving scheduling and memory allocation onto the GPU itself.

## Project Structure

```
osproj/
├── Core Runtime (Required)
│   ├── grosr_runtime.h          # Unified GROSR runtime header
│   ├── grosr_allocator.cu       # GPU-side slab allocator
│   ├── grosr_queue.cu           # Task queue management
│   └── grosr_runtime.cu         # Persistent runtime kernel
│
├── Main Experiments (Current)
│   ├── exp_a_pingpong.cu        # Experiment A: Ping-Pong Latency
│   ├── exp_b_throughput.cu      # Experiment B: Throughput Benchmark
│   ├── exp_c_graph_bfs.cu      # Experiment C: Graph BFS Macrobenchmark
│   └── test_allocator.cu        # Allocator unit tests
│
├── Legacy Demos (Reference Only)
│   ├── benchmark_exp0.cu        # Original benchmark (superseded by exp_a/b)
│   ├── exp0_microkernel.cu       # Original microkernel demo (superseded by unified runtime)
│   └── exp1_syscalls.cu         # GPU syscall proxy demo (proof of concept)
│
├── Build & Scripts
│   ├── Makefile                 # Build system
│   └── run_experiments.py       # Legacy Python script (for benchmark_exp0.cu)
│
└── Documentation
    ├── README.md                # This file
    ├── RESEARCH_PLAN.md         # Detailed research plan
    ├── IMPLEMENTATION_SUMMARY.md # Implementation details
    ├── GEMINI_CHAT.md          # Historical reference (chat logs)
    └── GPT_CHAT.md             # Historical reference (chat logs)
```

## Building

### Prerequisites
- NVIDIA GPU with CUDA support (sm_70+)
- CUDA Toolkit 11.0+
- Python 3.x with matplotlib (for plotting)

### Build Commands

```bash
# Build main experiments (recommended)
make main

# Build all executables (including legacy demos)
make all

# Build for specific architecture (e.g., A100)
make ARCH=sm_80 main

# Build specific experiment
make exp_a_pingpong
make exp_b_throughput
make exp_c_graph_bfs

# Run tests
make test

# Run benchmarks
make benchmark

# Clean build artifacts
make clean
```

## Main Experiments

### Experiment A: Ping-Pong Latency
Measures the time to launch execution of a dependent task.

**Baseline**: CPU launches Kernel 1 → CPU reads → CPU launches Kernel 2  
**GROSR**: GPU Kernel 1 → GPU Scheduler → GPU Kernel 2 (no CPU)

```bash
./exp_a_pingpong [num_tasks]
```

**Expected Result**: 5-20× reduction in latency (0.5μs vs 10μs)

### Experiment B: Throughput Benchmark
Measures throughput with many small tasks, includes statistical analysis.

**Baseline**: Standard CUDA kernel launches in loop  
**GROSR**: Persistent kernel with task queue

```bash
./exp_b_throughput [num_tasks] [iterations]
```

**Expected Result**: 10-50× improvement in throughput

### Experiment C: Graph BFS Macrobenchmark
Demonstrates real-world benefit in dynamic, irregular workload (graph traversal).

**Baseline**: CPU checks queue size → launches kernel → GPU processes → CPU checks again  
**GROSR**: GPU threads add neighbors to queue dynamically → GPU scheduler processes autonomously

```bash
./exp_c_graph_bfs [num_nodes] [edge_probability] [source_node]
```

**Expected Result**: 1.2-2× end-to-end speedup with reduced CPU overhead

### Allocator Test
Tests GPU-side slab allocator functionality.

```bash
./test_allocator
```

### Experiment D: Allocator Microbenchmark
Compares GROSR `gpu_malloc/gpu_free` against CUDA **device-side** `malloc/free`.

```bash
./exp_d_allocator_bench [num_threads] [iters_per_thread] [size_bytes] [mode] [outstanding] [touch_bytes]
```

## Key Components

### GPU-Side Slab Allocator
- **Location**: `grosr_allocator.cu`
- **Functions**: `gpu_malloc()`, `gpu_free()`
- **Size Classes**: 32B, 64B, 128B, 256B, 512B, 1KB, 2KB, 4KB
- **Purpose**: Enable dynamic memory allocation entirely on GPU

### Persistent Runtime Kernel
- **Location**: `grosr_runtime.cu`
- **Function**: `grosr_runtime_kernel()`
- **Purpose**: Scheduler loop that processes tasks from queue without CPU intervention

### Task Queue
- **Location**: `grosr_queue.cu`
- **Functions**: `init_task_queue()`, `push_task()`, `pop_task()`
- **Purpose**: CPU-GPU IPC via Unified Memory ring buffer

## Usage Example

```cpp
#include "grosr_runtime.h"

// Initialize
TaskQueue q;
init_task_queue(&q, 1024, sizeof(SimpleTask));

volatile int* stop_flag;
cudaMallocManaged((int**)&stop_flag, sizeof(int));
*stop_flag = 0;

GROSRRuntime runtime;
runtime.task_queue = q;
runtime.stop_flag = stop_flag;
runtime.results = d_results;

// Launch persistent runtime
grosr_runtime_kernel<<<1, 1>>>(runtime);

// Push tasks from CPU
SimpleTask task;
task.task_id = 0;
task.data = 42;
push_task(&q, task);

// Cleanup
*stop_flag = 1;
cudaDeviceSynchronize();
cleanup_task_queue(&q);
```

## Research Plan

See `RESEARCH_PLAN.md` for:
- Detailed research objectives
- Experimental design
- Timeline and milestones
- Related work analysis

## Status

### Phase 1: Foundation ✅ COMPLETED
- [x] Unified GROSR runtime header
- [x] Persistent GPU microkernel
- [x] CPU-GPU communication mechanisms
- [x] Basic benchmarking infrastructure

### Phase 2: Core Components 🔄 IN PROGRESS
- [x] GPU-side slab allocator
- [x] Experiment A (ping-pong latency)
- [x] Experiment B (throughput with statistics)
- [~] Experiment C (macrobenchmark - Graph BFS prototype; needs validation/tuning)

## Writing the Report / Reproducing Results

The main report lives in `PROJECT.md`. To collect numbers for tables/plots:

```bash
make clean
make ARCH=sm_70 main    # or sm_80 for A100, sm_90 for H100
make test
./exp_a_pingpong 1000
./exp_b_throughput 10000 10
./exp_c_graph_bfs 10000 0.01 0
```

Each experiment prints CSV-style output that can be pasted into your report tables/plots.

## Contributing

This is a research project. For questions or issues, refer to the research plan or contact the project maintainer.

## License

Research project - see course guidelines.

