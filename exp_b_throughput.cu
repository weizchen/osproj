// Experiment B: Throughput Benchmark (Improved version)
// Measures throughput with many small tasks, includes statistical analysis

#include "grosr_runtime.h"
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>

// Baseline: Standard CUDA kernel launches
__global__ void baseline_task_kernel(int* results, int task_id) {
    results[task_id] = task_id * 2;
}

void run_baseline_throughput(int num_tasks, int num_iterations) {
    int* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, num_tasks * sizeof(int)));
    
    std::vector<double> times;
    times.reserve(num_iterations);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_tasks; i++) {
            baseline_task_kernel<<<1, 1>>>(d_results, i);
            cudaDeviceSynchronize(); // Must sync to measure true overhead
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }
    
    // Calculate statistics
    std::sort(times.begin(), times.end());
    double mean = 0;
    for (double t : times) mean += t;
    mean /= times.size();
    
    double median = times[times.size() / 2];
    double min_time = times[0];
    double max_time = times[times.size() - 1];
    
    // Standard deviation
    double variance = 0;
    for (double t : times) {
        variance += (t - mean) * (t - mean);
    }
    double stddev = sqrt(variance / times.size());
    
    double throughput = (num_tasks * 1000.0) / mean; // Tasks per second
    
    printf("Baseline_Throughput,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f\n",
           num_tasks, mean, median, min_time, max_time, stddev, throughput);
    
    cudaFree(d_results);
}

// GROSR: Persistent kernel with task queue
// SimpleTask is already defined in grosr_runtime.h
// We use grosr_runtime_kernel directly

void run_grosr_throughput(int num_tasks, int num_iterations) {
    int* d_results;
    CUDA_CHECK(cudaMallocManaged(&d_results, num_tasks * sizeof(int)));
    
    std::vector<double> times;
    times.reserve(num_iterations);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        TaskQueue q;
        // Avoid producer backpressure dominating the measurement.
        // Keep a minimum size for small runs, but allow large queues for large N.
        int capacity = num_tasks + 64;
        if (capacity < 1024) capacity = 1024;
        init_task_queue(&q, capacity, sizeof(SimpleTask));
        
        volatile int* stop_flag;
        CUDA_CHECK(cudaMallocManaged((int**)&stop_flag, sizeof(int)));
        *stop_flag = 0;
        
        GROSRRuntime runtime;
        runtime.task_queue = q;
        runtime.stop_flag = stop_flag;
        runtime.results = d_results;
        runtime.allocator = nullptr;
        
        // Launch persistent runtime
        // Use a full block so the runtime can process tasks in parallel.
        grosr_runtime_kernel<<<1, 256>>>(runtime);
        CUDA_CHECK(cudaGetLastError());
        
        // Warmup
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // IMPORTANT: Do not call cudaDeviceSynchronize() here; the runtime kernel is persistent.
        // Synchronizing the device would wait for the persistent kernel to exit (it won't until stop_flag=1).
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Push all tasks
        for (int i = 0; i < num_tasks; i++) {
            while (__atomic_load_n(q.head, __ATOMIC_RELAXED) - __atomic_load_n(q.tail, __ATOMIC_RELAXED) >= q.capacity) {}
            
            SimpleTask task;
            task.task_id = i;
            task.data = i;
            push_task(&q, task);
        }
        
        // Wait for completion
        while (*(volatile int*)q.tail < num_tasks) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
        
        // Cleanup for this iteration
        *stop_flag = 1;
        cudaDeviceSynchronize();
        cleanup_task_queue(&q);
        cudaFree((int*)stop_flag);
    }
    
    // Calculate statistics
    std::sort(times.begin(), times.end());
    double mean = 0;
    for (double t : times) mean += t;
    mean /= times.size();
    
    double median = times[times.size() / 2];
    double min_time = times[0];
    double max_time = times[times.size() - 1];
    
    double variance = 0;
    for (double t : times) {
        variance += (t - mean) * (t - mean);
    }
    double stddev = sqrt(variance / times.size());
    
    double throughput = (num_tasks * 1000.0) / mean; // Tasks per second
    
    printf("GROSR_Throughput,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f\n",
           num_tasks, mean, median, min_time, max_time, stddev, throughput);
    
    cudaFree(d_results);
}

int main(int argc, char** argv) {
    int num_tasks = 10000;
    int num_iterations = 10;
    
    if (argc > 1) num_tasks = atoi(argv[1]);
    if (argc > 2) num_iterations = atoi(argv[2]);
    
    printf("Experiment B: Throughput Benchmark (with statistics)\n");
    printf("Method,Tasks,Mean_ms,Median_ms,Min_ms,Max_ms,StdDev_ms,Throughput_ops_sec\n");
    
    run_baseline_throughput(num_tasks, num_iterations);
    run_grosr_throughput(num_tasks, num_iterations);
    
    return 0;
}

