# Research Plan: GROSR - Eliminating the Control-Plane Wall with a GPU-Resident Runtime

## Executive Summary

This research project investigates **GROSR (GPU-Resident Operating System Runtime)**, a focused system that eliminates the CPU control-plane bottleneck for fine-grained, dynamic GPU workloads. Rather than building a general-purpose GPU OS, GROSR moves the **control plane** (scheduling and memory allocation) onto the GPU itself, enabling autonomous computation without CPU intervention. The project targets a **>10× speedup** for dynamic workloads like agentic AI, graph processing, and irregular simulations where kernel launch overhead dominates execution time.

**Scope**: Dynamic Scheduling + Dynamic Memory + One Killer App Benchmark  
**Target Venue**: HotOS, HotCloud, or OSDI/ATC workshop/poster  
**Timeline**: OS Course Project (1 semester)

---

## 1. Research Hypothesis

### Core Hypothesis
**Current GPU utilization is bound by the latency of the CPU control plane** (launching kernels, checking status, managing memory). By moving the control plane (scheduling & memory allocation) onto the GPU itself, we can achieve **>10× speedups** for fine-grained, dynamic workloads (like Agentic AI or Graph Processing).

### Key Insight
The problem is not data movement (bandwidth) but **control latency**. Even with unified memory and fast interconnects, the CPU must still issue commands, creating a "control-plane wall" that limits GPU autonomy.

### Scope Boundaries
**What GROSR IS:**
- A GPU-resident runtime for **computation scheduling** (not I/O offloading)
- Dynamic task dispatch without CPU wake-ups
- GPU-side memory allocation for dynamic data structures
- Autonomous agent loops running entirely on GPU

**What GROSR IS NOT:**
- A general-purpose OS for GPU (too broad for course project)
- A data-plane bypass (GPUfs/GPUNet already do this)
- A static optimization (CUDA Graphs already exists)
- Hardware-level changes (focus on software runtime)

---

## 2. Background and Motivation

### The Control-Plane Wall

Modern GPU execution follows a **host-device model**:
1. CPU launches kernel → GPU executes → CPU checks completion → CPU launches next kernel
2. Each round-trip incurs **5-30μs latency** (kernel launch overhead)
3. For fine-grained workloads (many small kernels), this overhead dominates

**Example**: An agentic AI loop that generates tokens and decides next action:
- Baseline: CPU checks token → CPU launches kernel → GPU generates → CPU checks → repeat
- GROSR: GPU scheduler checks token → GPU launches function → GPU generates → GPU decides → repeat (no CPU)

### Why Existing Solutions Fall Short

#### 1. The "I/O Offloaders" (GPUfs, GPUNet, Zephyr)
- **What they did**: Allowed GPUs to talk to storage/network directly
- **Limitation**: Still rely on CPU OS for metadata and heavy lifting. They are **data plane** bypasses, not **control plane** replacements
- **Our angle**: GROSR moves the **decision making** (control plane) to the GPU

#### 2. The "Static Optimizers" (CUDA Graphs)
- **What they did**: Record a graph of kernels, driver replays it fast
- **Limitation**: It is **static**. Cannot change graph based on data (e.g., "If token is 'STOP', end loop")
- **Our angle**: GROSR is **dynamic**. GPU scheduler can branch, loop, and decide what to run next based on runtime data

#### 3. The "Hardware Solutions" (NVIDIA Grace-Hopper/Unified Memory)
- **What they did**: Made memory coherent between CPU and GPU
- **Limitation**: CPU still has to issue commands. Latency still governed by PCIe/C2C interconnects
- **Our angle**: Hardware fixes bandwidth, but GROSR fixes **latency**

### Research Gap

**No existing system provides GPU-resident control-plane scheduling** for dynamic computation. Prior work either:
- Offloads I/O (data plane) but keeps control on CPU
- Optimizes static execution graphs but cannot adapt dynamically
- Improves hardware but doesn't eliminate software control overhead

---

## 3. Research Questions

1. **RQ1**: Can a GPU-resident scheduler eliminate CPU control-plane latency, achieving >10× reduction in task dispatch time for fine-grained workloads?

2. **RQ2**: Can a GPU-side slab allocator enable dynamic memory allocation without host calls, supporting irregular data structures (trees, graphs) entirely in VRAM?

3. **RQ3**: Do autonomous GPU agent loops (graph traversal, token generation) achieve measurable speedups when control decisions happen on-GPU vs. CPU-mediated?

---

## 4. System Design: GROSR Architecture

### 4.1 The Persistent "Kernel-in-Kernel"

Instead of CPU launching `kernel_A` then `kernel_B`, CPU launches **one persistent kernel**: `GROSR_Runtime`.

**Properties**:
- Never finishes (runs until job is done)
- Occupies 1 SM (Streaming Multiprocessor)
- Runs `while(true)` loop monitoring Task Queue
- Uses CUDA Dynamic Parallelism (CDP) or function pointers to dispatch work

**Key Insight**: This eliminates `cudaLaunchKernel` from the critical path.

### 4.2 The Memory Allocator (The Hard Part)

**Problem**: Cannot call `cudaMalloc` inside a kernel.

**Solution**: Implement a **Slab Allocator** on GPU:
- CPU pre-allocates a large chunk (e.g., 4GB) of global memory (Arena)
- GROSR divides this into "Slabs" (32B, 64B, 1KB blocks)
- GPU threads perform atomic operations to "allocate" and "free" these blocks locally
- Enables dynamic data structures (Trees, Graphs) entirely on VRAM

**Why This Matters**: Enables irregular workloads (graph BFS, dynamic trees) without CPU round-trips.

### 4.3 The "Context Switch" (Dynamic Function Dispatch)

**How to run different "programs" without CPU?**

**Solution**: Function Pointers
- Store device function pointers in an array
- Scheduler reads Task ID → looks up `Function_Pointer_Array[ID]` → executes it
- Requires Dynamic Parallelism (CDP) or device-side function calls

### Architecture Diagram

```
┌─────────────────────────────────────────────┐
│           CPU Host Process                  │
│  - Initializes GROSR Runtime                │
│  - Pre-allocates Memory Arena               │
│  - Pushes initial tasks to queue            │
└───────────────┬─────────────────────────────┘
                │
        Unified Memory (Shared Queue)
                │
┌───────────────▼─────────────────────────────┐
│         GROSR Runtime (Persistent Kernel)   │
│         Running on 1 SM                      │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │  Scheduler Loop (while true)         │   │
│  │  1. Check Task Queue                 │   │
│  │  2. Dispatch via Function Pointer    │   │
│  │  3. Manage Memory (Slab Allocator)  │   │
│  └──────────────────────────────────────┘   │
│                                              │
│  ┌──────────────┬──────────────────────┐    │
│  │ Task Queue    │ Memory Arena         │    │
│  │ (Ring Buffer) │ (Slab Allocator)     │    │
│  └──────────────┴──────────────────────┘    │
└──────────────────────────────────────────────┘
```

---

## 5. Experimental Design

### Experiment A: The "Ping-Pong" Latency (Microbenchmark)

**Goal**: Measure time to launch execution of a dependent task.

**Baseline**:
```
CPU launches Kernel 1 (writes X) 
→ CPU reads X (cudaDeviceSynchronize) 
→ CPU launches Kernel 2
```

**GROSR**:
```
GPU Kernel 1 writes X 
→ GPU Scheduler sees X (checks queue) 
→ GPU Scheduler calls Function 2 (via function pointer)
```

**Metrics**:
- Latency per task dispatch (μs)
- CPU wake-ups avoided

**Expected Result**: GROSR should be **5-20× faster** (0.5μs vs 10μs)

**Status**: ✅ Partially implemented (`benchmark_exp0.cu`)

---

### Experiment B: Throughput under Contention (Microbenchmark)

**Goal**: Measure throughput with many small tasks.

**Scenario**: 10,000 tiny tasks (e.g., simple arithmetic operations)

**Baseline**: `cudaLaunchKernel` in a loop (with synchronization)

**GROSR**: Push 10,000 task structs into Ring Buffer, GPU scheduler processes autonomously

**Metrics**:
- Tasks per second (throughput)
- Total execution time
- CPU utilization

**Visualization**: Bar chart showing "Tasks Per Second" (Baseline vs GROSR)

**Expected Result**: GROSR should achieve **10-50× higher throughput**

**Status**: ✅ Implemented (`benchmark_exp0.cu`, `run_experiments.py`)

---

### Experiment C: The Macrobenchmark (The "Why")

**Goal**: Demonstrate real-world benefit in dynamic, irregular workload.

**Candidate Workloads**:

**Option 1: Graph Breadth-First Search (BFS)**
- **Why**: Irregular. Some nodes have 1 neighbor, some have 1000. Requires dynamic queue management.
- **Baseline**: CPU checks queue size → launches kernel with size N → GPU processes → CPU checks again
- **GROSR**: GPU threads add neighbors to queue dynamically → GPU scheduler keeps eating from queue until empty → No CPU interaction
- **Metrics**: Time to complete BFS, CPU-GPU round-trips

**Option 2: Autonomous Agent Loop (Token Generation)**
- **Why**: Agent decides next action based on current state (dynamic control flow)
- **Baseline**: CPU checks token → CPU launches kernel → GPU generates → CPU checks → repeat
- **GROSR**: GPU scheduler checks token → GPU launches function → GPU generates → GPU decides → repeat
- **Metrics**: Tokens per second, end-to-end latency

**Option 3: Monte Carlo Simulation**
- **Why**: Dynamic branching based on random outcomes
- **Baseline**: CPU manages simulation state, launches kernels per iteration
- **GROSR**: GPU manages state, scheduler dispatches work dynamically
- **Metrics**: Simulations per second, CPU overhead

**Expected Result**: **1.2-2× end-to-end speedup** (modest but significant, with much larger CPU overhead reduction)

**Status**: ⏳ Not started (recommend Graph BFS as most "OS-like")

---

## 6. Implementation Status

### Completed Components ✅

1. **Experiment 0: GPU-Resident Microkernel**
   - Persistent kernel with task queue (`exp0_microkernel.cu`, `experiment0_persistent_runtime.cu`)
   - CPU-GPU IPC via Unified Memory ring buffer
   - Basic task processing loop

2. **Experiment 1: Benchmark Framework**
   - Baseline vs GROSR comparison (`benchmark_exp0.cu`)
   - Python automation script (`run_experiments.py`)
   - Latency and throughput measurement

3. **Experiment 2: GPU Syscall Proxy** (Proof of Concept)
   - GPU-initiated syscalls (`exp1_syscalls.cu`)
   - CPU proxy thread handling I/O
   - Demonstrates GPU-CPU communication pattern

### In Progress 🔄

1. **Experiment A Refinement**: Complete ping-pong latency measurements
2. **Experiment B Analysis**: Comprehensive throughput benchmarking

### Planned ⏳

1. **GPU-Side Slab Allocator** (Critical for Experiment C)
   - Implement `gpu_malloc()` and `gpu_free()` using atomic operations
   - Support multiple slab sizes (32B, 64B, 1KB)
   - Benchmark against `cudaMallocAsync`

2. **Dynamic Function Dispatch**
   - Function pointer array for task dispatch
   - Integration with scheduler loop

3. **Experiment C: Macrobenchmark**
   - Implement Graph BFS with GROSR
   - Compare against baseline CPU-orchestrated version

---

## 7. Paper Outline

### Title
**GROSR: Eliminating the Control-Plane Wall with a GPU-Resident Runtime**

### Structure

#### 1. Introduction (1-2 pages)
- **Hook**: GPUs are fast, but "feeding" them is slow
- **Problem**: The "Control-Plane Wall." As kernels get smaller (AI Agents), launch overhead dominates execution time
- **Solution**: GROSR. Move the OS kernel into the GPU
- **Contributions**: 
  - First GPU-resident control-plane scheduler
  - >10× reduction in task dispatch latency
  - GPU-side memory allocator enabling dynamic workloads

#### 2. Background & Motivation (1-2 pages)
- Standard CUDA execution model (Host-Device)
- Diagram of "Round Trip" latency
- Why CUDA Graphs isn't enough (it's static)
- Why I/O offloaders (GPUfs) don't solve control-plane problem

#### 3. Design (2-3 pages)
- **3.1 The Microkernel**: Persistent thread structure
- **3.2 Memory Management**: GPU-side Slab Allocator (include diagram)
- **3.3 Scheduling**: Ring Buffer and warp-level dispatching
- **3.4 Dynamic Dispatch**: Function pointers and CDP

#### 4. Implementation (1-2 pages)
- "We implemented GROSR in C++ and CUDA 12..."
- Challenges: 
  - Deadlocks (if scheduler waits for blocked worker)
  - Memory Consistency (using `__threadfence_system()`)
  - Atomic operations for allocator

#### 5. Evaluation (2-3 pages)
- **Exp A**: Latency (ping-pong benchmark)
- **Exp B**: Throughput (10,000 tasks)
- **Exp C**: Graph Traversal (BFS) or Agent Loop
- Comparison with baseline CUDA and CUDA Graphs

#### 6. Related Work (1 page)
- GPUfs, GPUNet, Zephyr (distinguish: they focus on I/O, we focus on computation scheduling)
- CUDA Graphs (distinguish: static vs dynamic)
- Hardware solutions (distinguish: bandwidth vs latency)

#### 7. Conclusion (0.5 pages)
- "GROSR proves that GPUs are ready to be autonomous compute agents."
- Future work: Multi-tenant scheduling, fault tolerance

---

## 8. Literature Review & Positioning

### Key Papers to Read

1. **GPUfs (ASPLOS 2013)** - Closest intellectual ancestor
   - GPU-initiated file I/O
   - **Our distinction**: They focus on data plane, we focus on control plane

2. **Hornet (Graph Processing on GPU)**
   - Dynamic graph algorithms on GPU
   - **Our distinction**: They use CPU for scheduling, we move scheduling to GPU

3. **GENESYS (OSDI 2018)**
   - GPU-initiated syscalls
   - **Our distinction**: They virtualize CPU syscalls, we eliminate CPU from control path

4. **CUDA Dynamic Parallelism**
   - Kernels launching kernels
   - **Our distinction**: CDP still has overhead, GROSR uses persistent kernel

### Positioning Statement

GROSR is the **first system to provide GPU-resident control-plane scheduling** for dynamic computation. Unlike prior work that offloads I/O (data plane) or optimizes static graphs, GROSR moves decision-making onto the GPU, eliminating CPU round-trips for fine-grained workloads.

---

## 9. Timeline and Milestones

### Phase 1: Foundation (Weeks 1-4) ✅ COMPLETED
- [x] Implement persistent GPU microkernel
- [x] Establish CPU-GPU communication mechanisms
- [x] Create basic benchmarking infrastructure
- [x] Run initial latency/throughput experiments

### Phase 2: Core Components (Weeks 5-8) 🔄 IN PROGRESS
- [x] Complete Experiment A (ping-pong latency)
- [x] Complete Experiment B (throughput benchmarking)
- [ ] **Implement GPU-Side Slab Allocator** ← **CURRENT PRIORITY**
- [ ] Implement dynamic function dispatch
- [ ] Benchmark allocator vs `cudaMallocAsync`

### Phase 3: Macrobenchmark (Weeks 9-12)
- [ ] Implement Graph BFS with GROSR
- [ ] Implement baseline CPU-orchestrated BFS
- [ ] Run comprehensive comparison
- [ ] Collect all experimental data

### Phase 4: Analysis and Writing (Weeks 13-16)
- [ ] Analyze all experimental results
- [ ] Write paper draft (Introduction, Design, Evaluation)
- [ ] Create figures and diagrams
- [ ] Related work section
- [ ] Final revisions and submission

---

## 10. Success Criteria

### Minimum Viable Results
- ✅ Demonstrate persistent GPU microkernel processing tasks
- ✅ Show measurable reduction in task dispatch latency (target: **5× minimum**)
- ⏳ Implement working GPU-side memory allocator

### Target Results
- **>10× reduction** in task dispatch latency (Experiment A)
- **10-50× improvement** in throughput (Experiment B)
- **1.2-2× end-to-end speedup** in macrobenchmark (Experiment C)
- GPU-side allocator achieving **~10× faster** small-object allocation

### Stretch Goals
- Multi-tenant GPU execution with GPU-side scheduling
- Fault tolerance mechanisms
- Publication-quality results suitable for HotOS/HotCloud or OSDI/ATC workshop

---

## 11. Technical Challenges and Mitigation

### Challenge 1: GPU Memory Coherence
**Issue**: Ensuring CPU-GPU memory consistency for shared data structures (task queue, allocator metadata).

**Mitigation**: 
- Use Unified Memory with explicit synchronization (`__threadfence_system()`)
- Atomic operations for queue head/tail pointers
- Careful memory ordering in allocator

### Challenge 2: Deadlock Prevention
**Issue**: If scheduler waits for a worker that is blocked, system deadlocks.

**Mitigation**: 
- Timeout mechanisms in scheduler
- Non-blocking queue operations
- Careful design of wait conditions

### Challenge 3: Slab Allocator Complexity
**Issue**: Implementing efficient GPU-side allocator with atomic operations is non-trivial.

**Mitigation**: 
- Start with simple bitmap allocator
- Use atomic operations (`atomicCAS`, `atomicExch`)
- Benchmark incrementally

### Challenge 4: Debugging Persistent Kernels
**Issue**: Debugging persistent kernels is more difficult than standard kernels.

**Mitigation**: 
- Extensive logging (GPU printf)
- CPU-side monitoring tools
- Incremental development and testing

---

## 12. Next Immediate Steps

### This Week (Priority Order)

1. **Implement GPU-Side Slab Allocator** ⭐ **HIGHEST PRIORITY**
   - Design data structures (arena, slab headers, free lists)
   - Implement `gpu_malloc(size)` and `gpu_free(ptr)` device functions
   - Test with simple allocation patterns
   - Benchmark against `cudaMallocAsync`

2. **Complete Experiment A Analysis**
   - Run ping-pong benchmark multiple times
   - Collect statistical data (mean, median, percentiles)
   - Create visualization

3. **Design Experiment C (Graph BFS)**
   - Choose graph representation (CSR format)
   - Design task structure for BFS
   - Plan baseline implementation

### Next 2 Weeks

1. Implement dynamic function dispatch mechanism
2. Integrate allocator with scheduler
3. Begin Experiment C implementation

---

## 13. Resources and Dependencies

### Hardware Requirements
- NVIDIA GPU (sm_70+): V100, T4, RTX 20xx+, A100, H100
- CUDA Toolkit 11.0+ (for Dynamic Parallelism support)
- Unified Memory support

### Software Dependencies
- CUDA runtime
- Python 3.x with matplotlib, numpy
- Standard C++ compiler

### External Resources
- **Key Papers**: GPUfs (ASPLOS '13), Hornet (graph processing), GENESYS (OSDI '18)
- CUDA documentation and best practices
- GPU architecture references (NVIDIA programming guide)

---

## 14. Conclusion

This research plan outlines a **focused, achievable** investigation into GPU-resident control-plane scheduling. By narrowing the scope to **dynamic scheduling + dynamic memory + one killer app**, we avoid the trap of trying to build "Linux for GPU" while still making a meaningful contribution.

The project builds on established foundations (persistent kernels, unified memory) and targets a clear, falsifiable hypothesis: **moving the control plane to GPU eliminates CPU round-trip latency, achieving >10× speedups for fine-grained workloads**.

**Current Status**: Foundation established, core microbenchmarks completed, ready to implement GPU-side allocator and macrobenchmark.

**Key Differentiator**: GROSR focuses on **computation scheduling** (control plane), not I/O offloading (data plane), making it complementary to prior work while addressing an unexplored research gap.

---

*Last Updated: Based on narrowed scope from Gemini chat*  
*Repository: `/homes/c/chenw274/osproj`*  
*Target Venue: HotOS, HotCloud, or OSDI/ATC workshop*
