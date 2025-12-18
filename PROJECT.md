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

### Validation methodology (applies to all experiments)
To ensure baseline and GROSR are computing the same thing (and we are not “benchmarking a bug”), each experiment prints explicit correctness checks outside the timed region:

- **Exp A**: validates `data[i] == 2*i + 1` for all tasks.
- **Exp B**: validates `results[i] == 2*i` for all tasks (GPU-side mismatch counter).
- **Exp C (chain graphs)**: validates deterministic BFS outputs (`dist[i]=i`, `parent[i]=i-1` for source=0).
- **Exp D**: focuses on allocator throughput; correctness is separately validated by `test_allocator`.

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

#### 6.3.1 Experiment B+ (fair baselines): sync vs batched vs bulk vs GROSR
The original Exp B baseline (`exp_b_throughput`) is intentionally **worst-case** (launch+sync per task) to expose the “control-plane wall”.
To provide fairer context, `exp_b_plus_fair` adds additional CPU baselines:

- **Baseline_Sync**: launch one kernel per task and synchronize after every launch (worst-case).
- **Baseline_Batched**: launch one kernel per task but synchronize only once at the end (removes per-task sync, still pays per-task launch).
- **Baseline_Bulk**: one kernel computes all results (best-case when tasks are uniform and fusible; represents what you would do if you can avoid “many tiny kernels” entirely).
- **Baseline_Graphs**: capture the `N` tiny-kernel launch sequence once as a CUDA Graph and replay it (useful when the task DAG is static and repeatable).
- **GROSR_Throughput**: persistent kernel + queue (GPU-resident control plane).

**Commands**:
- `make exp_b_plus_fair`
- `./exp_b_plus_fair <num_tasks> <iters>`

**Results (RTX 3070, CUDA 11.8; 5 iterations)**:

| Tasks (N) | Sync (ops/s) | Batched (ops/s) | Graphs (ops/s) | Bulk (ops/s) | GROSR (ops/s) |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 222,265 | 692,023 | 1,207,201 | 149,494,708 | 1,255,771 |
| 10,000 | 222,263 | 705,856 | 1,224,808 | 1,204,354,947 | 5,806,131 |
| 50,000 | 218,160 | 778,227 | 1,380,123 | 5,742,901,773 | 16,437,864 |

**Interpretation**:
- GROSR provides large gains over **Sync** and **Batched** baselines (i.e., when the workload genuinely consists of many small tasks that cannot be fused easily).
- **Graphs** reduces CPU launch overhead when the kernel DAG is static, but it still replays the same `N` tiny-kernel launches; GROSR can still win by batching and processing tasks inside a persistent kernel.
- **Bulk** is an important counterpoint: if tasks are uniform and independent, fusing into a single kernel can outperform both GROSR and per-task launches. This helps scope GROSR’s value proposition to *dynamic/irregular/unfusible* workloads where kernel fusion/graphs are not straightforward.

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

### 6.6 Experiment D: Allocator microbenchmark
**Question**: How fast is GROSR’s GPU-side slab allocator compared to CUDA device `malloc/free` for small allocations?

This experiment runs `alloc+touch+free` loops inside a GPU kernel and reports allocation throughput. The benchmark exposes a `touch_bytes` knob: touching more bytes per allocation makes the workload more realistic and typically reduces allocator-dominated speedups (eventually becoming bandwidth-bound).

**Fairness / how to interpret the speedup**: This is a **specialized vs. general-purpose** comparison. GROSR’s allocator is a slab allocator restricted to small sizes (≤4KB) backed by a pre-allocated arena, while CUDA device `malloc/free` is a general-purpose heap designed to handle many cases (with higher metadata/locking overhead). Therefore, large speedups (often \(10^2\times\) in highly concurrent small-allocation workloads) are plausible and do not imply GROSR is universally better. Also note that the benchmark only “touches” a few bytes per allocation to isolate allocator overhead; workloads that heavily initialize/use allocated memory will become more bandwidth/compute-bound and see a smaller end-to-end advantage.

**Results (RTX 3070, CUDA 11.8; 65,536 threads × 50 iters/thread)**:

| Size (bytes) | GROSR allocs/s | Device malloc allocs/s | Speedup (GROSR / Device) |
|---:|---:|---:|---:|
| 32 | 582,960,893 | 5,090,305 | 114.5× |
| 64 | 647,130,589 | 5,429,308 | 119.2× |
| 256 | 648,183,335 | 3,912,934 | 165.6× |
| 1024 | 565,071,512 | 1,774,620 | 318.4× |

**Interpretation**: For small allocations, GROSR’s slab allocator is orders of magnitude faster than CUDA’s device `malloc/free`, which is consistent with the design goal (fast, GPU-local small-object allocation). This supports using `gpu_malloc/gpu_free` for dynamic data structures in GPU-resident runtimes.

**More realistic workload modes**: To avoid a “best-case” microbenchmark, `exp_d_allocator_bench` also supports modes with **outstanding allocations** and **mixed size classes** (simulating object lifetimes and size variability).
Below is one representative setting (RTX 3070, CUDA 11.8; 16,384 threads × 200 iters/thread; base size 64B; outstanding=16):

| Mode | Description | GROSR allocs/s | Device malloc allocs/s | Speedup | GROSR success | Device success |
|---:|---|---:|---:|---:|---:|---:|
| 0 | churn (alloc+free immediately) | 283,024,600 | 3,075,601 | 92.0× | 1.0000 | 1.0000 |
| 1 | outstanding (delayed frees) | 249,571,044 | 1,219,893 | 204.6× | 1.0000 | 1.0000 |
| 2 | mixed sizes (64–512B mix) | 35,701,885 | 178,597 | 199.9× | 0.9772 | 0.9775 |

**Note**: Mode 2 can exhibit allocation failures because the current slab allocator partitions the arena into fixed per-size-class pools; under skewed or mixed distributions, a size class can run out even if total arena space remains. We report **success rate** to make this behavior explicit.

**Practical takeaway**: Exp D supports the claim that a GPU-resident runtime benefits from a fast small-object allocator, but it also highlights a real limitation (fixed size-class partitioning). A production-quality allocator would add rebalancing between size classes, per-SM caches, and/or fallback paths for large or skewed allocations.

**Commands**:
- `make exp_d_allocator_bench`
- `./exp_d_allocator_bench [num_threads] [iters_per_thread] [size_bytes] [mode] [outstanding] [touch_bytes]`

**Speedup range with increased memory touch**: In mode=1 (outstanding allocations), base size 256B, outstanding=16, we observe that increasing `touch_bytes` reduces GROSR’s raw allocation-throughput advantage because the benchmark becomes more dominated by memory writes. For example at 2048 threads × 200 iters:

| TouchBytes | GROSR allocs/s | Device malloc allocs/s | Speedup |
|---:|---:|---:|---:|
| 4 | 39,778,112 | 247,131 | 160.9× |
| 64 | 17,632,019 | 246,953 | 71.4× |
| 256 | 6,639,741 | 240,487 | 27.6× |

## 7. Discussion
### 7.1 What worked
- Persistent scheduling + a shared queue is enough to demonstrate CPU control-plane overhead in microbenchmarks.
- A simple device-side allocator enables dynamic allocation patterns without host calls.

### 7.2 Limitations
- **Single consumer / single-scheduler thread**: limits scaling and can become a bottleneck.
- **Fixed queue capacity**: backpressure via spinning is simple but not ideal.
- **No preemption**: long-running tasks can starve others.
- **Not a true OS**: no isolation, interrupts, or syscall execution on the GPU.

**Strong baselines / when GROSR is unnecessary**: If the workload is **static and repeatable**, CUDA **Graphs** can significantly reduce CPU overhead by replaying a captured launch DAG. If tasks are also **uniform and independent**, the best solution is often **kernel fusion / a single bulk kernel** (our “Baseline_Bulk”), which can outperform both GROSR and multi-launch baselines. GROSR is most compelling when the workload is **dynamic, irregular, or data-dependent**, where building a fixed graph or a fused kernel is difficult.

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