// GROSR Runtime Kernel Implementation

#include "grosr_runtime.h"

// Persistent runtime kernel (scheduler loop)
__global__ void grosr_runtime_kernel(GROSRRuntime runtime) {
    // Single-block cooperative scheduler: process up to blockDim.x tasks per iteration.
    // This keeps the runtime simple while removing the "single-thread scheduler" bottleneck.
    int tid = threadIdx.x;
    if (blockIdx.x != 0) return;

    __shared__ int s_tail;
    __shared__ int s_claim;

    while (atomicAdd_system((int*)runtime.stop_flag, 0) == 0) {
        int head = atomicAdd_system(runtime.task_queue.head, 0);
        int tail = atomicAdd_system(runtime.task_queue.tail, 0);
        int avail = head - tail;

        if (avail <= 0) {
            if (tid == 0) __nanosleep(1000);
            continue;
        }

        if (tid == 0) {
            s_tail = tail;
            s_claim = (avail < (int)blockDim.x) ? avail : (int)blockDim.x;
        }
        __syncthreads();

        int claim = s_claim;
        if (tid < claim) {
            int idx = s_tail + tid;
            int slot = idx % runtime.task_queue.capacity;
            char* base = (char*)runtime.task_queue.tasks + (slot * runtime.task_queue.task_size);

            size_t task_size = runtime.task_queue.task_size;
            if (task_size == sizeof(SimpleTask)) {
                SimpleTask task = *(SimpleTask*)base;
                if (runtime.results) runtime.results[task.task_id] = task.data * 2;
            } else if (task_size == sizeof(Task)) {
                Task task = *(Task*)base;
                int result = 0;
                if (task.op == 0) result = task.a + task.b;
                else if (task.op == 1) result = task.a * task.b;
                if (runtime.results) runtime.results[task.result_idx] = result;
            }
        }

        // Publish results before advancing tail (single consumer).
        __threadfence_system();
        __syncthreads();
        if (tid == 0) atomicAdd_system(runtime.task_queue.tail, claim);
    }
}

