// GROSR Runtime - GPU-Resident Operating System Runtime
// Unified header for GROSR components

#ifndef GROSR_RUNTIME_H
#define GROSR_RUNTIME_H

#include <cuda_runtime.h>
#include <cstdio>

// ============================================================================
// CUDA Error Checking Macro
// ============================================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(1); \
        } \
    } while (0)

// ============================================================================
// Task Queue Structures
// ============================================================================

// Simple task for benchmarking
struct SimpleTask {
    int task_id;
    int data;
};

// Task with operation type (for more complex workloads)
struct Task {
    int op;           // 0 = add, 1 = mul, etc.
    int a, b;
    int result_idx;   // index into results[]
};

// Ring buffer task queue
struct TaskQueue {
    void* tasks;      // Generic pointer (can be SimpleTask* or Task*)
    int   capacity;
    int*  head;       // Producer index (CPU)
    int*  tail;       // Consumer index (GPU)
    size_t task_size; // Size of each task in bytes
};

// ============================================================================
// GPU-Side Slab Allocator
// ============================================================================

// Slab size classes (power of 2 for simplicity)
#define SLAB_SIZE_32B   32
#define SLAB_SIZE_64B   64
#define SLAB_SIZE_128B  128
#define SLAB_SIZE_256B  256
#define SLAB_SIZE_512B  512
#define SLAB_SIZE_1KB   1024
#define SLAB_SIZE_2KB   2048
#define SLAB_SIZE_4KB   4096

#define NUM_SLAB_CLASSES 8

// Slab header: tracks free blocks in a slab
struct SlabHeader {
    unsigned int free_mask;  // Bitmask: 1 = free, 0 = allocated
    int next_free;           // Index of next free block (or -1)
    int num_blocks;           // Number of blocks in this slab
    int block_size;           // Size of each block
};

// Slab allocator state
struct SlabAllocator {
    char* arena;              // Pre-allocated memory arena
    size_t arena_size;        // Total size of arena
    
    SlabHeader* slab_headers; // Array of slab headers
    int num_slabs_per_class[NUM_SLAB_CLASSES];
    int slab_sizes[NUM_SLAB_CLASSES];
    
    // Free lists: one per size class
    int* free_lists[NUM_SLAB_CLASSES];  // Array of free slab indices per class
    int* free_list_tops[NUM_SLAB_CLASSES]; // Top of each free list
};

// Initialize slab allocator (called from CPU)
void init_slab_allocator(SlabAllocator* alloc, size_t arena_size);

// Cleanup slab allocator
void cleanup_slab_allocator(SlabAllocator* alloc);

// Helper: Find size class for a given size
__device__ __host__ int find_size_class(size_t size);

// GPU-side allocation function (implementation in grosr_allocator.cu)
// Forward declaration - implementation must be visible at compile time
__device__ void* gpu_malloc(SlabAllocator* alloc, size_t size);

// GPU-side free function (implementation in grosr_allocator.cu)
__device__ void gpu_free(SlabAllocator* alloc, void* ptr);

// ============================================================================
// GROSR Persistent Runtime Kernel
// ============================================================================

// Runtime state
struct GROSRRuntime {
    TaskQueue task_queue;
    SlabAllocator* allocator;  // Optional: GPU-side allocator
    volatile int* stop_flag;
    int* results;              // Optional: result array
};

// Persistent runtime kernel (scheduler loop)
__global__ void grosr_runtime_kernel(GROSRRuntime runtime);

// ============================================================================
// Queue Management Functions
// ============================================================================

// Initialize task queue
void init_task_queue(TaskQueue* q, int capacity, size_t task_size);

// Cleanup task queue
void cleanup_task_queue(TaskQueue* q);

// CPU: Push task to queue (template implementation)
template<typename T>
void push_task(TaskQueue* q, const T& task) {
    // Wait for space (simple backpressure)
    while (*(volatile int*)q->head - *(volatile int*)q->tail >= q->capacity) {
        // Yield to avoid busy-waiting
    }
    
    int slot = (*(volatile int*)q->head) % q->capacity;
    T* task_ptr = (T*)((char*)q->tasks + (slot * q->task_size));
    *task_ptr = task;
    
    __sync_synchronize(); // Memory barrier
    *(volatile int*)q->head = *(volatile int*)q->head + 1;
}

// GPU: Check queue size
__device__ __forceinline__ int queue_size(const TaskQueue& q) {
#if __CUDA_ARCH__ >= 600
    int head = atomicAdd_system(q.head, 0);  // Atomic load with system scope
    int tail = atomicAdd_system(q.tail, 0);  // Atomic load with system scope
#else
    int head = atomicAdd(q.head, 0);
    int tail = atomicAdd(q.tail, 0);
#endif
    return head - tail;
}

// GPU: Pop task from queue (returns true if task was popped)
__device__ __forceinline__ bool pop_task(TaskQueue* q, void* task_out, size_t task_size) {
    // Load current head and tail
#if __CUDA_ARCH__ >= 600
    int head = atomicAdd_system(q->head, 0);  // Atomic load with system scope
    int tail = atomicAdd_system(q->tail, 0);  // Atomic load with system scope
#else
    int head = atomicAdd(q->head, 0);
    int tail = atomicAdd(q->tail, 0);
#endif
    
    if (tail >= head) {
        return false; // Queue empty
    }
    
    // Copy task data before incrementing tail
    int slot = tail % q->capacity;
    char* task_src = (char*)q->tasks + (slot * q->task_size);
    
    // Copy using simple loop (memcpy not available in device code)
    for (size_t i = 0; i < task_size; i++) {
        ((char*)task_out)[i] = task_src[i];
    }
    
    // Atomically increment tail to claim this task
    __threadfence_system();
    
#if __CUDA_ARCH__ >= 600
    atomicAdd_system(q->tail, 1);
#else
    atomicAdd(q->tail, 1);
#endif
    
    return true;
}

#endif // GROSR_RUNTIME_H
