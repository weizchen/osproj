# GROSR: A GPU-Resident Operating System Runtime

**Author**: Weizhe Chen — University of Toronto  
**Date**: December 2025

## Abstract

Modern GPUs deliver massive throughput, but fine-grained and dynamic workloads face a "control-plane wall": the latency and CPU overhead of frequent kernel launches, host-side polling, and dynamic memory management often dominate execution time. This project presents **GROSR**, a minimal GPU-resident runtime that moves scheduling and allocation logic to the device. By using a persistent kernel, a Unified Memory task queue, and a device-side slab allocator, GROSR enables the GPU to autonomously fetch tasks and manage memory. We evaluate GROSR against optimized CPU baselines. Our results show that GROSR achieves **~9x higher throughput** than an optimized batched CPU baseline for small tasks and outperforms CUDA `malloc` by **>100x** for small allocations. We also demonstrate a BFS prototype that executes entirely on the GPU, highlighting both the potential and current limitations of this approach.

---

## 1. Introduction

As GPU workloads shift from monolithic data-parallel kernels to dynamic, irregular patterns (e.g., graph algorithms, ray tracing, agent simulations), the host CPU often becomes a bottleneck. The round-trip latency of launching a kernel, coupled with the serialization of managing GPU memory from the host, creates a "control-plane wall" that limits performance for fine-grained tasks.

This project investigates the hypothesis: **Can moving a thin slice of OS-like responsibility (dispatch and allocation) to the GPU significantly reduce these overheads?**

We implement and evaluate **GROSR** (GPU-Resident Operating System Runtime), which provides:
1.  **Persistent Scheduling**: A long-running kernel that keeps the GPU active, eliminating per-task launch latency.
2.  **Autonomous Memory Management**: A fast, device-side slab allocator (`gpu_malloc`/`gpu_free`).
3.  **Low-Latency Dispatch**: A ring-buffer queue in Unified Memory (UVM) for producer-consumer interaction.

---

## 2. System Design

GROSR is designed not as a full general-purpose OS, but as a high-performance substrate for dynamic GPU applications.

### 2.1 Architecture
The runtime consists of three core components:

*   **Persistent Runtime Kernel**: This kernel is launched once and loops purely on the GPU. It polls a task queue, executes work, and can dynamically spawn new work without CPU involvement.
*   **Unified Memory Queue**: A lock-free-style ring buffer facilitates communication. The CPU (or other GPU threads) pushes tasks, and the persistent runtime consumes them.
*   **Slab Allocator**: To support dynamic data structures (like graphs or trees) without the heavy overhead of `cudaMalloc`, GROSR uses a pre-allocated memory arena divided into fixed-size slabs (32B to 4KB). Allocation is an O(1) bitmap scan in the common case.

### 2.2 Improvements & Optimization
During development, we identified key bottlenecks and implemented optimizations:
*   **Allocator Search Hints**: Originally an O(N) search, we optimized the allocator to cache the last known free slab, bringing allocation closer to O(1).
*   **Warp-Aggregated Queue**: For the BFS workload, we implemented warp-level aggregation to reduce atomic contention on the queue head/tail pointers.

---

## 3. Evaluation

We evaluated GROSR on an NVIDIA RTX 3070 (Ampere) using CUDA 11.8. We devised four experiments to test specific aspects of the runtime.

### Experiment A: Latency (Ping-Pong)
*   **Goal**: Measure the minimum latency to dispatch a dependent task.
*   **Method**: A "Ping-Pong" chain of dependent tasks.
*   **Result**: GROSR achieves **~5.5 µs** per task dispatch, compared to **~9.5 µs** for standard CPU launches.
*   **Conclusion**: Moving the scheduler to the GPU reduces dispatch latency by **~40%** by avoiding the driver stack for every dependency.

### Experiment B: Throughput (The "Fair" Benchmark)
*   **Goal**: Measure maximum task throughput for small, independent tasks.
*   **Methodology**: To ensure a fair comparison, we tested against *multiple* high-quality baselines:
    1.  **Baseline (Sync)**: Naive launch-per-task + sync (strawman).
    2.  **Baseline (Batched)**: Asynchronous launches + single sync (optimized CPU baseline).
    3.  **GROSR**: Persistent runtime.
*   **Results (N=10,000 tasks)**:
    *   **Baseline (Sync)**: ~208,000 ops/sec
    *   **Baseline (Batched)**: ~705,000 ops/sec
    *   **GROSR**: **~6,525,000 ops/sec**
*   **Analysis**: GROSR is **~9.2x faster** than the optimized "Batched" CPU baseline. Even when the CPU avoids synchronization, the sheer driver overhead of issuing commands limits throughput. The persistent kernel eliminates this bottleneck.

### Experiment C: Irregular Workloads (Graph BFS)
*   **Goal**: Validate capability for dynamic, self-feeding workloads.
*   **Method**: A Breadth-First Search (BFS) where visiting a node dynamically enqueues neighbors.
*   **Findings**:
    *   **Functionality**: GROSR successfully executes the full BFS traversal without CPU intervention derived from frontier logic.
    *   **Performance**: The prototype is currently slower (~0.5x speedup) than a CPU-managed level-synchronous approach for simple dense graphs.
    *   **Bottleneck**: The single-threaded scheduler in the current prototype cannot saturate the GPU memory bandwidth as effectively as a massive grid launch. This highlights the need for *multi-consumer* scheduling (which we began implementing via warp-aggregation).

### Experiment D: Allocator Microbenchmark
*   **Goal**: Compare GROSR's allocator against CUDA's `device malloc`.
*   **Method**: A "churn" test where thousands of threads allocate, touch, and free memory.
*   **Results**:
    *   **CUDA Device Malloc**: ~1.5 Million allocations/sec
    *   **GROSR Slab Allocator**: **~297 Million allocations/sec**
*   **Analysis**: For small, fixed-size objects (64B), GROSR is **~190x faster**. This confirms that a specialized, arena-based allocator is essential for high-performance dynamic GPU applications, as the general-purpose `malloc` allows too much contention and overhead.

---

## 4. Discussion and Future Work

### 4.1 Comparison with Baselines
We found that "fair" baselines are critical. A simple CPU loop is easy to beat, but an optimized batched launch (Exp B) or a pre-captured CUDA Graph (which we also implemented) offers stiff competition. GROSR wins specifically when workloads are **dynamic** (cannot be pre-graphed) and **fine-grained** (driver overhead dominates).

### 4.2 Limitations
*   **Single-Threaded Scheduler**: The primary bottleneck for Exp C is that one GPU thread manages the queue. Scaling this to a multi-warp or block-wide consumer is the most important next step.
*   **Security/Isolation**: GROSR provides no memory protection between tasks, making it suitable for a single trusted application but not as a multi-tenant OS.

### 4.3 Conclusion
This project demonstrates that a GPU-resident runtime is not only feasible but necessary for unleashing the performance of fine-grained dynamic workloads. By cutting the CPU out of the critical path, we unlocked nearly an order of magnitude in throughput and two orders of magnitude in allocation performance.

---

## 5. How to Reproduce
1.  **Build**: `make clean && make`
2.  **Run Throughput Test**: `./exp_b_plus_fair 50000 10`
3.  **Run Allocator Test**: `./exp_d_allocator_bench 65536 100 64 0 0 4`