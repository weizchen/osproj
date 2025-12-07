// benchmark_exp0.cu
// NVCC Compile: nvcc -O2 -arch=sm_70 benchmark_exp0.cu -o bench0

#include <cstdio>
#include <chrono>
#include <thread>
#include <vector>

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("ERR: %s\n", cudaGetErrorString(err)); exit(1); } }

// --- 1. BASELINE: Standard Kernel Launch ---
__global__ void baseline_kernel(int* data, int i) {
    // Do trivial work
    data[i] = i * 2;
}

void run_baseline(int num_tasks) {
    int* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, num_tasks * sizeof(int)));

    // Synchronize before starting timer
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_tasks; ++i) {
        baseline_kernel<<<1, 1>>>(d_data, i);
        // We must sync to measure true launch+overhead latency per task
        // In real apps, this sync happens implicitly via dependencies or explicit calls
        cudaDeviceSynchronize(); 
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Baseline,%d,%.3f,%.2f\n", num_tasks, ms, num_tasks / (ms / 1000.0));
    
    cudaFree(d_data);
}

// --- 2. GROSR: Persistent Kernel ---
struct TaskQueue {
    int* tasks; // simplified for benchmark: task is just an int index
    int* head;
    int* tail;
    int* results;
};

__global__ void persistent_kernel(TaskQueue q, volatile int* stop_flag) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;

    while (*stop_flag == 0) {
        int h = *q.head;
        int t = *q.tail;
        
        if (t < h) {
            // Process task
            int task_idx = q.tasks[t % 1024];
            q.results[task_idx] = task_idx * 2;
            
            // Mark done
            __threadfence_system();
            *q.tail = t + 1;
        }
    }
}

void run_persistent(int num_tasks) {
    // Setup
    TaskQueue q;
    CUDA_CHECK(cudaMallocManaged(&q.tasks, 1024 * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&q.head, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&q.tail, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&q.results, num_tasks * sizeof(int)));
    int* stop_flag;
    CUDA_CHECK(cudaMallocManaged(&stop_flag, sizeof(int)));
    
    *q.head = 0; *q.tail = 0; *stop_flag = 0;

    // Launch Runtime
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    persistent_kernel<<<1, 1, 0, stream>>>(q, stop_flag);
    
    // Warmup
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    cudaDeviceSynchronize(); // Sync host before timing
    auto start = std::chrono::high_resolution_clock::now();

    // Push Tasks
    for (int i = 0; i < num_tasks; ++i) {
        // Flow control
        while (*q.head - *q.tail >= 1024) {} // spin
        
        q.tasks[*q.head % 1024] = i;
        *q.head = *q.head + 1;
    }

    // Wait for completion
    while (*q.tail < num_tasks) {}

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    printf("GROSR,%d,%.3f,%.2f\n", num_tasks, ms, num_tasks / (ms / 1000.0));

    // Cleanup
    *stop_flag = 1;
    cudaDeviceSynchronize();
    cudaFree(q.tasks); cudaFree(q.head); cudaFree(q.tail); cudaFree(q.results); cudaFree(stop_flag);
}

int main(int argc, char** argv) {
    int num_tasks = 10000; // Default
    if (argc > 1) num_tasks = atoi(argv[1]);

    // CSV Header
    // printf("Method,Tasks,TotalTime_ms,Ops_Per_Sec\n");
    
    // Run Experiments
    run_baseline(num_tasks);
    run_persistent(num_tasks);
    
    return 0;
}