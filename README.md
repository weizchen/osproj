# GROSR: A GPU-Resident OS-like Runtime for Reducing CPU Control-Plane Overhead

ECE 1759 - Advances in Operating Systems

Weizhe Chen 1006669525

## Before the report
This project is inspired by classic OS/architecture co-design work: **“The Interaction of Architecture and Operating System Design”** by **Thomas E. Anderson, Henry M. Levy, Brian N. Bershad, and Edward D. Lazowska**. A central lesson is that OS structure and performance are shaped by the underlying hardware interface; when the hardware changes, OS design opportunities and bottlenecks change as well.

Today, GPUs and accelerators are increasingly central to real workloads, but they are still treated as “attached devices” controlled by a CPU-centric OS. Therefore I wanted to do a small, modern “OS + architecture” experiment. I believe **accelerators should become first-class citizens** of future systems software.

Modern GPUs deliver massive throughput, but fine-grained and dynamic GPU workloads can be bottlenecked by the **CPU control plane**: frequent kernel launches, host synchronization, CPU-side polling, and host-managed dynamic memory. This project explores a minimal “GPU-resident OS runtime” (**GROSR**) that consists of a persistent GPU scheduler, a CPU↔GPU task queue, and a device-side allocator that allow GPUs to execute many small tasks with reduced CPU intervention. Evaluation is done on designed microbenchmarks and macrobenchmarks that include kernel launch comparison, throughput microbenchmark, allocator benchmark, a graph BFS benchmark and a small ReLU+Softmax benchmark. The goal is to characterize where a GPU-resident control plane helps, where strong baselines exist, and where this approach has fundamental limitations. Although the benchmarks cannot faithfully represent the real-world workload, they serve as a starting point as a proof of concept and offer insights to future work.

## 1. Introduction
Modern GPUs are typically orchestrated by a host CPU for kernel dispatch and dynamic control. When work is expressed as many small kernels, CPU launch overhead and synchronization can become an expensive cost relative to the GPU computation. This project explores whether moving a small slice of OS-like control-plane logic onto the device can reduce that overhead for GPU work and improve performance.

GROSR is a runtime prototype rather than a general-purpose OS. It does not implement preemptive multitasking, isolation, or syscalls; instead, it provides a minimal substrate for GPU-side task dispatch and small-object allocation, and evaluates that substrate against various baselines such as batching, CUDA Graph, and kernel fusion, to clarify how and where GPU-resident control is helpful versus unnecessary.

## 2. Background and Motivation
### 2.1 Motivation and scope
Even with modern memory systems and fast interconnects, host-driven orchestration can add high fixed overhead relative to fine-grained GPU tasks, consuming CPU time and complicating coordination. At the same time, GPUs do not expose key OS mechanisms (privilege modes, traps/interrupts, cheap preemption), so GROSR targets a narrower and more realistic scope: **GPU-side scheduling and allocation for GPU computation**, while keeping initialization/termination and I/O CPU-mediated.

### 2.4 Related Work

This project builds on research elevating GPUs to first-class compute resources. **OS Abstractions**: Prior systems like **PTask** (SOSP'11) and **GPUfs** (ASPLOS'13) successfully moved dataflow management and file I/O services directly to the GPU, proving that OS-like functionality can exist on-device. GROSR complements these I/O-centric works by focusing specifically on the *control plane*—reducing the latency of task scheduling and execution without heavy driver intervention. **Persistent Execution**: To bypass the overhead of hardware kernel launches, **Gupta et al.** (HPEC'12) formalized "persistent threads," where a long-running kernel consumes work directly from memory. GROSR adopts this model for its soft-scheduler, trading hardware occupancy for nanosecond-scale dispatch latency. **Memory Allocation**: Throughput-oriented allocators like **DynaSOAr** (ASPLOS'19) and **Gelado & Garland** (PPoPP'19) use complex hierarchical synchronization to support millions of operations per second. In contrast, GROSR implements a lightweight bitmap-based slab allocator, prioritizing low latency for small, fixed-size objects over the massive scalability of these production-grade systems. **Baselines**: I evaluate GROSR against industry standard optimizations: **CUDA Graphs** and **Kernel Fusion**. This comparison highlights the specific regime—dynamic, irregular, data-dependent workloads—where runtime software scheduling provides value over static compile-time optimizations.


## 3. Design and Implementation
GROSR is a small research prototype. It provides no isolation/privilege separation and no external I/O subsystem; the CPU still initializes the runtime and terminates it via a stop flag. The prototype is built from three concrete components. These are currently limited in functionality to provide a simple functional demonstration:

- **Persistent scheduler kernel**: a long-running CUDA kernel that repeatedly pulls tasks from a queue and executes them on the GPU.
- **CPU↔GPU task queue**: a bounded ring buffer in Unified Memory used to publish tasks from the CPU to the GPU runtime.
- **GPU slab allocator**: a device-side allocator for small, fixed-size allocations to avoid `cudaMalloc`/`cudaFree` in the critical path.

### 3.1 Persistent scheduler kernel
At the core is `grosr_runtime_kernel`, which runs a scheduling loop until `stop_flag` is set. The baseline runtime in this repo uses a **single-block cooperative scheduler**: one block polls the queue, and up to `blockDim.x` threads cooperatively execute a batch of available tasks per loop iteration. When no work is available, it backs off briefly (via `__nanosleep`) to avoid a tight spin.

The runtime executes tasks inline in the persistent kernel rather than launching a new kernel per task. In the simple microbenchmarks, task “dispatch” is implemented by checking `task_size` and interpreting each queue slot as either a `SimpleTask` or `Task` struct; results are optionally written to a `results[]` array. The scheduler is cooperative and intentionally simple. It does not provide preemption or fairness policies while using the single-block design can become a potential bottleneck for workloads that need higher parallel task consumption.

### 3.2 Unified Memory task queue
The queue (`TaskQueue` in `grosr_runtime.h`) is a ring buffer stored in Unified Memory:

- **Layout**: a raw `tasks` buffer, a `capacity`, a `task_size`, and two counters `head` (producer) and `tail` (consumer).
- **Producer (CPU)**: `push_task()` writes the task payload into the next slot and then advances `head`. It uses a **release fence** (`__atomic_thread_fence(__ATOMIC_RELEASE)`) to ensure the task data is visible before publishing the updated `head`.
- **Consumer (GPU)**: `pop_task()` checks `head` and `tail` via system-scope atomics (`atomicAdd_system(..., 0)`), copies the task payload from the queue slot, then uses `__threadfence_system()` before advancing `tail` to publish consumption.

It is not a general-purpose system as it assumes fixed-size task records and uses polling rather than interrupts, but this simple design is sufficient for demonstrating CPU→GPU control-plane overheads.

### 3.3 GPU slab allocator
The slab allocator (in `grosr_allocator.cu`) is designed for **highly concurrent small allocations**. It allocates a fixed arena in Unified Memory and divides it into 4KB slabs. Each slab has a `SlabHeader` containing the size class and a bitmap of free blocks.

- **Size classes**: 32B, 64B, 128B, 256B, 512B, 1KB, 2KB, 4KB.
- **Bitmap**: small size classes have more than 32 blocks per slab, so the allocator uses a **128-bit bitmap** to represent up to 128 blocks in a 4KB slab.
- **Allocation**: `gpu_malloc()` scans slabs within the chosen size class (starting at a thread-dependent offset) and claims a free block by atomically clearing one bit.
- **Free**: `gpu_free()` computes the slab and block index for a pointer and atomically sets the corresponding bit.

The allocator has a fixed arena and fixed per-class slab distribution; it can fail allocations for a given size class and it also does not support allocations larger than 4KB but it is enough to showcase the dynamic data allocation on GPU-side in this prototype.

## 4. Evaluation
Several experiments are intentionally designed to isolate CPU control-plane overhead (kernel launch and synchronization) and therefore favor GROSR’s strengths. I include strong baselines (batched launches, CUDA Graph replay, and fused kernels) and also report workloads where GROSR does not outperform baselines, to avoid overgeneralizing microbenchmark results to end-to-end applications.

### 4.1 Experimental setup
All results in this report were collected on the following machine:

- **GPU**: NVIDIA GeForce RTX 3070 (`sm_86`)
- **CUDA toolkit**: CUDA 11.8 (`nvcc` 11.8.89)
- **NVIDIA driver**: 535.247.01
- **CPU**: Intel i7-11700 (8 cores / 16 threads)
- **OS**: Debian 12 (bookworm), kernel 6.1.0-37-amd64

### 4.2 Experiment A: Ping-Pong dispatch latency
**Question**: How expensive is a dependent “two-stage” operation when driven by CPU kernel launches vs. a GPU-resident scheduler?

Baseline: CPU launches kernel1, synchronizes, launches kernel2, synchronizes. GROSR: CPU enqueues stage-1 tasks; the persistent GPU runtime schedules stage-2 work without a per-task CPU launch.

**Results**:

| Tasks (N) | Baseline avg (µs / pair) | GROSR avg (µs / pair) | Speedup (Baseline / GROSR) |
|---:|---:|---:|---:|
| 100 | 9.922 | 12.014 | 0.83× |
| 1,000 | 9.330 | 5.633 | 1.66× |
| 10,000 | 9.214 | 5.087 | 1.81× |
| 50,000 | 9.199 | 4.826 | 1.91× |

**Interpretation**: For small \(N\), fixed overheads and scheduling noise dominate; at larger \(N\), GROSR converges to ~5µs per dependent pair vs. ~9µs for CPU launches, giving ~1.7–1.9× lower dispatch latency in steady state.


### 4.3 Experiment C: Throughput (fair baselines)
**Question**: When work is expressed as many small tasks, how does a GPU-resident runtime compare against realistic CPU orchestration strategies?

More baselines are reported in this experiment:
- Baseline_Sync (worst case: sync after every launch),
- Baseline_Batched (launch all tasks then sync once), 
- Baseline_Graphs (capture/replay a static launch DAG), 
- Baseline_Bulk (a single fused kernel; best case when fully fusible),
- GROSR (persistent runtime + queue).

**Results (ops/s)**:

| Tasks (N) | Sync | Batched | Graphs | Bulk | GROSR |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 222,265 | 692,023 | 1,207,201 | 149,494,708 | 1,255,771 |
| 10,000 | 222,263 | 705,856 | 1,224,808 | 1,204,354,947 | 5,806,131 |
| 50,000 | 218,160 | 778,227 | 1,380,123 | 5,742,901,773 | 16,437,864 |

**Interpretation**: GROSR improves over per-task CPU launch strategies (Sync/Batched) and can exceed Graph replay in this microbenchmark because it batches and executes tasks inside a persistent kernel rather than replaying (N) separate launches. However, Bulk remains the best-case baseline when the workload is fully fusible; this is an important reminder that GROSR is primarily targeting dynamic or irregular workloads where fusion/Graphs are harder to apply.


### 4.4 Experiment D: Allocator microbenchmark
**Question**: How fast is GROSR’s GPU-side slab allocator compared to CUDA device `malloc/free` for small allocations?

This experiment runs allocation loops inside a GPU kernel and reports allocation throughput. It is a component microbenchmark (allocator cost), not an end-to-end application. I also report success rate because the current allocator uses a fixed arena with fixed per-size-class slab partitioning, so it may fail under skewed mixes even when some arena memory remains.

The configuration for the representative results below is: **RTX 3070, CUDA 11.8; 16,384 threads × 200 iters/thread; base size 64B; outstanding=16**.

The table below reports three workload modes: (0) immediate free (“churn”), (1) delayed frees with a fixed number of outstanding allocations, and (2) a mixed-size workload (64–512B) to expose fragmentation and partitioning effects.

**Results (allocs/s)**:

I report two baselines: CUDA device `malloc/free` and a more “production-like” alternative that allocates a **per-thread pool once** and then performs fast local suballocation (`ThreadPool_Suballoc`).

| Mode | Description | GROSR | Device malloc | ThreadPool_Suballoc | GROSR success | Device success | Pool success |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | churn (alloc+free immediately) | 154,477,431 | 6,087,899 | 1,148,188,002 | 1.0000 | 1.0000 | 1.0000 |
| 1 | outstanding (delayed frees) | 532,911,463 | 2,597,474 | 27,356,778 | 1.0000 | 1.0000 | 1.0000 |
| 2 | mixed sizes (64–512B mix) | 88,169,660 | 733,512 | 476,190,480 | 0.9847 | 1.0000 | 1.0000 |

**Interpretation**: GROSR’s slab allocator is much cheaper than a general-purpose device heap for highly concurrent small allocations, but **these numbers should not be read as end-to-end application speedups**. The `ThreadPool_Suballoc` baseline shows that if allocations are private and memory demand is bounded, applications can often get most of the benefit by preallocating per-thread scratch space; in contrast, shared or hard-to-bound allocation patterns require a global allocator. The mixed-size mode also highlights a limitation of the current GROSR allocator: fixed partitioning across size classes can lead to allocation failures even when total arena memory remains.


### 4.5 Experiment D: Graph BFS prototype
**Question**: Can GPU-side scheduling support irregular, dynamic work creation with reduced CPU orchestration?

This prototype compares a baseline that iteratively launches BFS kernels per frontier (one launch per level) against a GROSR version where the GPU runtime consumes task and enqueues neighbor work dynamically.

**Results**:

| Nodes (N) | Edges | Baseline time (ms) | GROSR time (ms) | Speedup (Baseline / GROSR) |
|---:|---:|---:|---:|---:|
| 5,000 | 9,998 | 54.980 | 152.385 | 0.36× |
| 10,000 | 19,998 | 101.256 | 173.763 | 0.58× |
| 20,000 | 39,998 | 202.183 | 234.316 | 0.86× |

**Interpretation**: GROSR correctly completes BFS without CPU “level-by-level” launches, but the current BFS runtime is still a **single-thread consumer** with heavy per-edge queue/atomic overhead, so it does not outperform the baseline yet. This macrobenchmark is still valuable in the report because it demonstrates the capability of GPU-side work creation and completion detection and clearly identifies the next step improvement: scalable GPU work distribution like multi-warp consumers.

### 4.6 Experiment E: ReLU + Softmax pipeline

**Question**: When each “task” is a small but real ML-style computation (ReLU + Softmax on a short vector), how do CPU launch strategies compare against a GPU-resident persistent runtime?


I compare Baseline_Batched (many tiny launches, one sync), Baseline_Graphs (replay a captured DAG when static), Baseline_Bulk (fully fused best case), and GROSR (persistent runtime with no per-task CPU launches). Baseline_Sync is included only as a “worst case” illustration.

**Results (ms)**:

| Method | Mean time |
|---|---:|
| Baseline_Batched | 68.925 |
| Baseline_Graphs | 52.392 |
| GROSR | 7.303 |
| Baseline_Bulk | 0.940 |

**Interpretation**: GROSR reduces control-plane overhead relative to “many tiny launches” (even with Graph replay), but the fused **Bulk** kernel remains the best-case baseline when the workload is fully fusible. This supports a realistic conclusion: GROSR’s value is in autonomy and control-plane reduction, not beating fused kernels on static workloads.

## 5. Discussion
### 5.1 What worked
At a prototype level, a persistent scheduler plus a shared queue is enough to demonstrate the CPU control-plane costs of fine-grained GPU work. A simple device-side allocator also enables dynamic allocation patterns without host calls, which is a prerequisite for more OS-like GPU runtimes.

### 5.2 Limitations
This prototype has major limitations. The current design often relies on a single consumer/scheduler thread, which limits scaling and can become a bottleneck. The queue is bounded and uses simple spinning for backpressure. There is no preemption, so long-running tasks can starve others. Finally, this is not a true OS: there is no isolation, interrupt/trap model, or syscall/I/O execution on the GPU.

If the workload is static and repeatable, CUDA Graph replay can reduce CPU overhead by replaying a captured launch DAG. If tasks are also uniform and independent, kernel fusion is often the best solution. GROSR is most compelling when the workload is dynamic, irregular, or data-dependent, where building a fixed graph or fused kernel is difficult.

### 5.3 Future work
Future work should focus on scaling the scheduler (multi-warp / multi-consumer queue designs), exploring simple scheduling policies, improving the allocator (per-SM caches, reduced contention, support of skewed size mixes and larger allocations), and using more realistic macrobenchmarks in evaluation.

## 6. Conclusion
GROSR shows that a small, GPU-resident runtime can reduce CPU control-plane overhead for fine-grained GPU workloads. The prototype demonstrates feasibility and highlights the architectural gaps that prevent a full GPU OS today.

---

## Appendix: Build & Reproducibility

### Build prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit (project tested on CUDA 11.8)

### Build commands
```bash
# Clean
make clean

# Build main experiments (recommended)
make ARCH=sm_86 main

# Build all executables
make ARCH=sm_86 all

# Build individual targets
make ARCH=sm_86 exp_a_pingpong
make ARCH=sm_86 exp_b_throughput
make ARCH=sm_86 exp_b_plus_fair
make ARCH=sm_86 exp_c_graph_bfs
make ARCH=sm_86 exp_d_allocator_bench
make ARCH=sm_86 exp_e_relu_softmax

# Run tests
make ARCH=sm_86 test
```

### Sample testcase command
```bash
./exp_a_pingpong 1000
./exp_b_plus_fair 10000 5
./exp_c_graph_bfs 10000 0.01 0 chain
./exp_d_allocator_bench 16384 200 64 1 16 4
./exp_e_relu_softmax 5000 256 2
```


