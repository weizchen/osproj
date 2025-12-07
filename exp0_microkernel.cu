// Compile with: nvcc -O2 -arch=sm_70 exp0_microkernel.cu -o exp0
// (Adjust -arch=sm_XX to match your GPU, e.g., sm_80 for A100)

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while(0)

// --- 1. Shared Data Structures (The "OS" State) ---

// A simple task definition (add or multiply)
struct Task {
    int op;         // 0 = ADD, 1 = MUL
    int a, b;
    int result_idx; // Where to write the result
};

// A ring buffer queue in Unified Memory
struct TaskQueue {
    Task* tasks;
    int   capacity;
    int* head;     // CPU writes here
    int* tail;     // GPU reads here
};

// --- 2. The GPU "Microkernel" (Persistent Runtime) ---

__global__ void gpu_os_runtime(TaskQueue q, int* results, volatile int* stop_flag) {
    // Only one thread (scheduler) runs the loop. 
    // In a real OS, this would dispatch work to other warps/blocks.
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;

    printf("[GPU] OS Runtime started on SM.\n");

    // Main Scheduler Loop
    while (*stop_flag == 0) {
        // Snapshot head/tail (atomic loads preferred in strict memory models, 
        // but volatile works for simple UM/PCIe coherence on x86/NVIDIA)
        int current_head = *q.head;
        int current_tail = *q.tail;

        if (current_tail == current_head) {
            // Queue is empty: Sleep to save power/heat (requires sm_70+)
            __nanosleep(1000); 
            continue;
        }

        // Process the task
        int slot = current_tail % q.capacity;
        Task t = q.tasks[slot];

        int val = 0;
        if (t.op == 0)      val = t.a + t.b;
        else if (t.op == 1) val = t.a * t.b;

        // Write result
        results[t.result_idx] = val;

        // Mark task as done by advancing tail
        // __threadfence_system() ensures CPU sees the result before seeing the tail update
        __threadfence_system();
        *q.tail = current_tail + 1;
    }

    printf("[GPU] Stop flag received. OS Runtime shutting down.\n");
}

// --- 3. CPU Mediator (The Host Process) ---

int main() {
    // Initialize Queue in Unified Memory
    int capacity = 1024;
    TaskQueue q;
    CUDA_CHECK(cudaMallocManaged(&q.tasks, capacity * sizeof(Task)));
    CUDA_CHECK(cudaMallocManaged(&q.head, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&q.tail, sizeof(int)));
    *q.head = 0; 
    *q.tail = 0;

    // Results and Flag
    int num_tasks = 100;
    int* results;
    int* stop_flag;
    CUDA_CHECK(cudaMallocManaged(&results, num_tasks * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&stop_flag, sizeof(int)));
    *stop_flag = 0;

    // Launch the GPU Runtime (Persistent Kernel)
    // Run on stream 0 (default), non-blocking
    gpu_os_runtime<<<1, 32>>>(q, results, stop_flag);
    CUDA_CHECK(cudaGetLastError());
    
    printf("[CPU] GPU Runtime launched. Pushing tasks...\n");

    // Push tasks to the running GPU kernel
    for (int i = 0; i < num_tasks; ++i) {
        // Simple backpressure: wait if queue is full
        while (*q.head - *q.tail >= capacity) {
            std::this_thread::yield();
        }

        int slot = *q.head % capacity;
        q.tasks[slot].op = (i % 2 == 0) ? 0 : 1; // Alternate Add/Mul
        q.tasks[slot].a = i;
        q.tasks[slot].b = 2;
        q.tasks[slot].result_idx = i;

        // Publish task
        *q.head = *q.head + 1;
        
        // Small sleep to simulate realistic arrival times (optional)
        if (i % 10 == 0) std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    // Wait for GPU to drain queue
    while (*q.tail < num_tasks) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Stop the GPU runtime
    *stop_flag = 1;
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    int errors = 0;
    for (int i = 0; i < num_tasks; ++i) {
        int expected = (i % 2 == 0) ? (i + 2) : (i * 2);
        if (results[i] != expected) {
            printf("Error at %d: Expected %d, Got %d\n", i, expected, results[i]);
            errors++;
        }
    }

    if (errors == 0) printf("[CPU] Success! All %d tasks processed by persistent GPU kernel.\n", num_tasks);
    
    // Cleanup
    cudaFree(q.tasks); cudaFree(q.head); cudaFree(q.tail);
    cudaFree(results); cudaFree(stop_flag);
    return 0;
}