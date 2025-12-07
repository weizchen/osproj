// Experiment A: Ping-Pong Latency Benchmark
// Measures time to launch execution of a dependent task

#include "grosr_runtime.h"
#include <chrono>
#include <thread>

// Baseline: CPU launches Kernel 1 -> CPU reads X -> CPU launches Kernel 2
__global__ void kernel1_baseline(int* data, int idx) {
    data[idx] = idx * 2;
}

__global__ void kernel2_baseline(int* data, int idx) {
    data[idx] = data[idx] + 1;
}

void run_baseline_pingpong(int num_tasks) {
    int* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, num_tasks * sizeof(int)));
    
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_tasks; i++) {
        // Launch kernel 1
        kernel1_baseline<<<1, 1>>>(d_data, i);
        cudaDeviceSynchronize(); // Wait for completion
        
        // Launch kernel 2 (dependent on kernel 1)
        kernel2_baseline<<<1, 1>>>(d_data, i);
        cudaDeviceSynchronize(); // Wait for completion
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_latency_us = (ms * 1000.0) / num_tasks; // Per task pair
    
    printf("Baseline_PingPong,%d,%.3f,%.3f\n", num_tasks, ms, avg_latency_us);
    
    cudaFree(d_data);
}

// GROSR: GPU Kernel 1 writes X -> GPU Scheduler sees X -> GPU Scheduler calls Function 2
struct PingPongTask {
    int task_id;
    int stage;  // 0 = stage1, 1 = stage2
};

// Stage 1 function
__device__ void process_stage1(int* data, int idx) {
    data[idx] = idx * 2;
}

// Stage 2 function  
__device__ void process_stage2(int* data, int idx) {
    data[idx] = data[idx] + 1;
}

// Enhanced task structure for ping-pong
// Note: This is different from SimpleTask, so we define it here
struct PingPongTaskEnhanced {
    int task_id;
    int stage;
    int* data_ptr;
};

__global__ void grosr_pingpong_runtime(TaskQueue* q, volatile int* stop_flag) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;
    
    while (*stop_flag == 0) {
        int head = atomicAdd(q->head, 0);
        int tail = atomicAdd(q->tail, 0);
        
        if (tail >= head) {
            __nanosleep(1000);
            continue;
        }
        
        // Pop task
        PingPongTaskEnhanced task;
        if (pop_task(q, &task, sizeof(PingPongTaskEnhanced))) {
            if (task.stage == 0) {
                process_stage1(task.data_ptr, task.task_id);
                // Immediately schedule stage 2
                PingPongTaskEnhanced stage2_task;
                stage2_task.task_id = task.task_id;
                stage2_task.stage = 1;
                stage2_task.data_ptr = task.data_ptr;
                
                // Push stage 2 back to queue (simulating GPU-side dispatch)
                int next_head = atomicAdd(q->head, 1);
                int slot = next_head % q->capacity;
                PingPongTaskEnhanced* task_ptr = (PingPongTaskEnhanced*)((char*)q->tasks + (slot * q->task_size));
                *task_ptr = stage2_task;
                __threadfence_system();
            } else {
                process_stage2(task.data_ptr, task.task_id);
            }
        }
    }
}

void run_grosr_pingpong(int num_tasks) {
    int* d_data;
    CUDA_CHECK(cudaMallocManaged(&d_data, num_tasks * sizeof(int)));
    
    TaskQueue* q;
    CUDA_CHECK(cudaMallocManaged(&q, sizeof(TaskQueue)));
    init_task_queue(q, 1024, sizeof(PingPongTaskEnhanced));
    
    volatile int* stop_flag;
    CUDA_CHECK(cudaMallocManaged((int**)&stop_flag, sizeof(int)));
    *stop_flag = 0;
    
    // Launch persistent runtime
    grosr_pingpong_runtime<<<1, 1>>>(q, stop_flag);
    CUDA_CHECK(cudaGetLastError());
    
    // Warmup
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    cudaDeviceSynchronize();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Push stage 1 tasks
    for (int i = 0; i < num_tasks; i++) {
        while (*q->head - *q->tail >= q->capacity) {}
        
        PingPongTaskEnhanced task;
        task.task_id = i;
        task.stage = 0;
        task.data_ptr = d_data;
        push_task(q, task);
    }
    
    // Wait for all tasks to complete (both stages)
    while (*(volatile int*)q->tail < num_tasks * 2) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_latency_us = (ms * 1000.0) / num_tasks; // Per task pair
    
    printf("GROSR_PingPong,%d,%.3f,%.3f\n", num_tasks, ms, avg_latency_us);
    
    // Cleanup
    *stop_flag = 1;
    cudaDeviceSynchronize();
    cleanup_task_queue(q);
    cudaFree(q);
    cudaFree(d_data);
    cudaFree((int*)stop_flag);
}

int main(int argc, char** argv) {
    int num_tasks = 1000;
    if (argc > 1) num_tasks = atoi(argv[1]);
    
    printf("Experiment A: Ping-Pong Latency Benchmark\n");
    printf("Method,Tasks,TotalTime_ms,AvgLatency_us\n");
    
    run_baseline_pingpong(num_tasks);
    run_grosr_pingpong(num_tasks);
    
    return 0;
}
