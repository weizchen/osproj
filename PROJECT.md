# GROSR: A GPU-Resident Runtime for Reducing CPU Control-Plane Overhead
Weizhe Chen — University of Toronto

## Abstract
Modern GPUs deliver massive throughput, but fine-grained and dynamic GPU workloads can be bottlenecked by the **CPU control plane**: frequent kernel launches, CPU-side polling, and host-managed dynamic memory. This project explores a minimal “GPU-resident OS runtime” *in the narrow sense of control-plane autonomy*: a persistent GPU scheduler, a CPU↔GPU task queue, and a device-side allocator that allow GPUs to execute many small tasks with reduced CPU intervention.

We implement **GROSR**, a prototype runtime built with CUDA persistent kernels and Unified Memory (UVM) communication. We evaluate GROSR using (A) a ping-pong dispatch microbenchmark, (B) a throughput microbenchmark with statistics, and (C) a graph BFS prototype representing irregular, dynamic work creation. Our results show that a persistent, GPU-side scheduler can reduce dispatch overhead for small tasks compared to CPU-launched baselines, while highlighting key limitations (single-SM scheduler, limited protection/isolation, and remaining CPU dependencies).

## 1. Introduction
GPUs remain tightly controlled by a host CPU for kernel dispatch and many forms of dynamic orchestration. For workloads made of many small kernels (e.g., dynamic graph traversals or agent-like pipelines), repeated host launches can dominate latency and CPU utilization. The core hypothesis of this project is:

- **Hypothesis**: Moving a small slice of OS-like control-plane logic (task dispatch and dynamic allocation) onto the GPU reduces overhead for fine-grained GPU work.

This project intentionally does **not** attempt to build a general-purpose OS on the GPU. Instead, it builds and evaluates a focused runtime substrate that makes GPU execution more autonomous in the common “many small tasks” setting.

## 2. Background and Motivation
### 2.1 The “control-plane wall”
Even with Unified Memory and fast interconnects, kernel launch and host synchronization introduce overheads that are *latency- and CPU-wakeup-heavy* relative to small GPU tasks.

### 2.2 Why not a full GPU OS?
GPUs lack classic OS mechanisms (privilege modes, interrupts/traps, fast preemption). Therefore, this project targets a pragmatic goal: **GPU-side scheduling and allocation for GPU computation**, while explicitly documenting what remains CPU-mediated.

## 3. System Overview
GROSR consists of three components:

- **Persistent scheduler kernel**: a long-lived GPU kernel that repeatedly pulls tasks from a queue and executes them.
- **CPU↔GPU task queue**: a ring buffer in Unified Memory used to publish tasks and consume completion.
- **GPU slab allocator**: a device-side allocator for small objects to support dynamic data structures without `cudaMalloc` in the critical path.

### 3.1 Threat model / scope
GROSR is a research prototype:

- No isolation, privilege separation, or security boundary.
- No external I/O subsystem or device driver model.
- CPU still initializes the runtime and can terminate it via a stop flag.

## 4. Design
### 4.1 Task model
Tasks are small structs placed in a ring buffer. The runtime supports a simple “task type by struct size” dispatch (prototype choice for simplicity).

### 4.2 Task queue design
The queue is a bounded ring buffer in Unified Memory with:

- CPU producer (`push_task`)
- GPU consumer (`pop_task`)

Correctness relies on explicit publish/consume ordering between host and device.

### 4.3 Allocator design
The allocator is a slab allocator with fixed 4KB slabs and per-slab bitmaps. Size classes: 32B–4KB. Allocation uses atomic bit operations to claim a free block.

## 5. Implementation
Implementation lives entirely in this repository:

- `grosr_runtime.cu`: persistent scheduler kernel
- `grosr_queue.cu` + `grosr_runtime.h`: queue initialization and push/pop helpers
- `grosr_allocator.cu`: GPU-side slab allocator (`gpu_malloc`, `gpu_free`)
- `test_allocator.cu`: allocator unit tests
- `exp_a_pingpong.cu`: ping-pong microbenchmark
- `exp_b_throughput.cu`: throughput benchmark with statistics
- `exp_c_graph_bfs.cu`: graph BFS prototype

## 6. Evaluation
### 6.1 Experimental setup (fill this in)
- **GPU**: <model, SM version>
- **CUDA toolkit**: <version>
- **CPU**: <model>
- **OS/driver**: <version>
- **Build flags**: `make ARCH=<sm_xx> main`

### 6.2 Experiment A: Ping-Pong dispatch latency
**Question**: How expensive is a dependent “two-stage” operation when driven by CPU kernel launches vs. a GPU-resident scheduler?

- Baseline: CPU launches kernel1, synchronizes, launches kernel2, synchronizes.
- GROSR: CPU enqueues stage-1 tasks; the persistent GPU runtime schedules stage-2 tasks.

**Results (RTX 3070, CUDA 11.8)**:

| Tasks (N) | Baseline avg (µs / pair) | GROSR avg (µs / pair) | Speedup (Baseline / GROSR) |
|---:|---:|---:|---:|
| 100 | 9.922 | 12.014 | 0.83× |
| 1,000 | 9.330 | 5.633 | 1.66× |
| 10,000 | 9.214 | 5.087 | 1.81× |
| 50,000 | 9.199 | 4.826 | 1.91× |

**Interpretation**: For small \(N\), fixed overheads and scheduling noise dominate; at larger \(N\), GROSR converges to ~5µs per dependent pair vs. ~9µs for CPU launches, giving ~1.7–1.9× lower dispatch latency in steady state.

**What to report**:
- Avg latency per task (µs) vs. number of tasks
- Optional: CPU utilization (even coarse) during the run

**Commands**:
- `make exp_a_pingpong`
- `./exp_a_pingpong <num_tasks>`

### 6.3 Experiment B: Throughput (many small tasks)
**Question**: What is the maximum rate of executing small tasks when CPU launch overhead is removed from the critical path?

**Results (RTX 3070, CUDA 11.8; 5 iterations)**:

| Tasks (N) | Baseline throughput (ops/s) | GROSR throughput (ops/s) | Speedup (GROSR / Baseline) |
|---:|---:|---:|---:|
| 1,000 | 220,424 | 1,333,603 | 6.05× |
| 5,000 | 220,191 | 5,962,224 | 27.07× |
| 10,000 | 220,525 | 5,833,240 | 26.45× |
| 50,000 | 231,207 | 15,904,217 | 68.79× |

**Interpretation**: After removing producer backpressure (queue sized to \(N\)) and parallelizing the persistent runtime to process tasks in batches across a full block, GROSR delivers substantially higher task throughput than CPU launches in this microbenchmark. The relatively large stddev at small \(N\) suggests fixed overheads/noise dominate short runs; the steady-state throughput reflects the benefit of avoiding per-task CPU launch/sync overhead.

**What to report**:
- Mean/median/min/max/stddev of iteration time
- Throughput (tasks/s)

**Commands**:
- `make exp_b_throughput`
- `./exp_b_throughput <num_tasks> <iters>`

### 6.4 Experiment C: Graph BFS prototype (macrobenchmark)
**Question**: Can GPU-side scheduling support irregular, dynamic work creation (neighbors push new tasks) with reduced CPU orchestration?

This prototype compares:
- Baseline: CPU iteratively launches BFS kernels per frontier.
- GROSR: GPU runtime consumes `BFSTask` and pushes neighbors as new tasks.

**What to report**:
- End-to-end BFS time (ms) for a few (nodes, edge_prob) settings
- Baseline: number of BFS iterations/levels (equals number of kernel launches)
- GROSR: nodes processed and termination condition reliability

**Commands**:
- `make exp_c_graph_bfs`
- `./exp_c_graph_bfs <num_nodes> <edge_prob> <source> [random|chain]`

**Results (RTX 3070, CUDA 11.8; chain graph)**:

The chain graph has \(\approx N\) BFS levels, so the baseline performs \(\approx N\) kernel launches (one per level), which stresses CPU control-plane overhead.

| Nodes (N) | Edges | Baseline time (ms) | GROSR time (ms) | Speedup (Baseline / GROSR) |
|---:|---:|---:|---:|---:|
| 5,000 | 9,998 | 54.980 | 152.385 | 0.36× |
| 10,000 | 19,998 | 101.256 | 173.763 | 0.58× |
| 20,000 | 39,998 | 202.183 | 234.316 | 0.86× |

**Interpretation**: GROSR correctly completes BFS (processes all nodes) without CPU “level-by-level” launches, but the current BFS runtime is still a **single-thread consumer** with heavy per-edge queue/atomic overhead, so it does not outperform the baseline yet. This macrobenchmark is still valuable in the report because it demonstrates the *capability* (GPU-side work creation and completion detection) and clearly identifies the next bottleneck: scalable GPU work distribution (multi-warp consumers / better frontier structure).

### 6.5 Allocator tests
**Goal**: show allocator correctness under concurrency.

**Commands**:
- `make test`
- `./test_allocator`

## 7. Discussion
### 7.1 What worked
- Persistent scheduling + a shared queue is enough to demonstrate CPU control-plane overhead in microbenchmarks.
- A simple device-side allocator enables dynamic allocation patterns without host calls.

### 7.2 Limitations
- **Single consumer / single-scheduler thread**: limits scaling and can become a bottleneck.
- **Fixed queue capacity**: backpressure via spinning is simple but not ideal.
- **No preemption**: long-running tasks can starve others.
- **Not a true OS**: no isolation, interrupts, or syscall execution on the GPU.

### 7.3 Future work (credible next steps)
- Multi-warp scheduler and/or multi-consumer queue
- Priority scheduling policies
- Better allocator: per-SM caches, reduced contention, larger allocations
- More realistic macrobenchmark (graph analytics dataset, not random graph)

## 8. Related Work (brief)
Situate GROSR relative to:
- GPUfs (GPU-initiated I/O with CPU mediation)
- GPUNet (GPU-driven networking)
- GENESYS (GPU-initiated syscalls via CPU)
- Zephyr/Glider (GPU-centric I/O paths)
- CUDA Graphs vs. dynamic runtime scheduling

## 9. Conclusion
GROSR shows that a small, GPU-resident runtime (persistent scheduler + UM queue + device allocator) can reduce CPU control-plane overhead for fine-grained GPU workloads. The prototype demonstrates feasibility and highlights the architectural gaps that prevent a full GPU OS today.

---

## Appendix: Reproducibility checklist
1. `make clean && make ARCH=<sm_xx> main`
2. `make test`
3. Run Exp A/B/C and paste CSV output into your report tables/plots.