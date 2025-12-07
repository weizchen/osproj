// GROSR Queue Management Implementation

#include "grosr_runtime.h"
#include <cstring>

// Initialize task queue
void init_task_queue(TaskQueue* q, int capacity, size_t task_size) {
    q->capacity = capacity;
    q->task_size = task_size;
    
    // Allocate queue buffer in unified memory
    CUDA_CHECK(cudaMallocManaged(&q->tasks, capacity * task_size));
    
    // Allocate head and tail pointers in unified memory
    CUDA_CHECK(cudaMallocManaged(&q->head, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&q->tail, sizeof(int)));
    
    *q->head = 0;
    *q->tail = 0;
}

// Cleanup task queue
void cleanup_task_queue(TaskQueue* q) {
    if (q->tasks) cudaFree(q->tasks);
    if (q->head) cudaFree(q->head);
    if (q->tail) cudaFree(q->tail);
}

// CPU: Push task to queue (template implementation in header)
// GPU functions are now inline in header

