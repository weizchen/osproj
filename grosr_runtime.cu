// GROSR Runtime Kernel Implementation

#include "grosr_runtime.h"

// Persistent runtime kernel (scheduler loop)
__global__ void grosr_runtime_kernel(GROSRRuntime runtime) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return; // Only thread 0 runs the scheduler
    
    while (atomicAdd((int*)runtime.stop_flag, 0) == 0) {
        // Check for tasks
        int head = atomicAdd(runtime.task_queue.head, 0);
        int tail = atomicAdd(runtime.task_queue.tail, 0);
        
        if (tail >= head) {
            // Queue empty: small backoff to save power
            __nanosleep(1000);
            continue;
        }
        
        // Process one task based on task size
        size_t task_size = runtime.task_queue.task_size;
        
        // For SimpleTask (8 bytes: 2 ints)
        if (task_size == sizeof(SimpleTask)) {
            SimpleTask task;
            if (pop_task(&runtime.task_queue, &task, sizeof(SimpleTask))) {
                // Process task
                if (runtime.results) {
                    runtime.results[task.task_id] = task.data * 2;
                }
            }
        }
        // For Task (16 bytes: 4 ints)
        else if (task_size == sizeof(Task)) {
            Task task;
            if (pop_task(&runtime.task_queue, &task, sizeof(Task))) {
                int result = 0;
                if (task.op == 0) {
                    result = task.a + task.b;
                } else if (task.op == 1) {
                    result = task.a * task.b;
                }
                
                if (runtime.results) {
                    runtime.results[task.result_idx] = result;
                }
            }
        }
        
        __threadfence_system(); // Ensure results are visible
    }
}

