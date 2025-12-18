// Experiment B+: Fair Throughput Benchmark
// Adds a "Batched" baseline to compare against GROSR.
//
// 1. Baseline (Sync): Launch + Sync per task (latency bound).
// 2. Baseline (Batched): Launch all + Sync once (throughput bound).
// 3. Baseline (Bulk): Launch ONE kernel to process all tasks (best-case CPU baseline if tasks are uniform/fusible).
// 4. Baseline (CUDA Graphs): Capture N tiny launches once, then replay (reduces launch overhead when graph is stable).
// 5. GROSR: Persistent kernel + Queue.

#include "grosr_runtime.h"
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>

// Baseline kernel
__global__ void baseline_task_kernel(int* results, int task_id) {
    results[task_id] = task_id * 2;
}

// Bulk baseline: one kernel computes all results
__global__ void baseline_bulk_kernel(int* results, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    results[tid] = tid * 2;
}

// Validator
__global__ void validate_double_results(const int* results, int n, int* mismatch_count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int expected = tid * 2;
    if (results[tid] != expected) atomicAdd(mismatch_count, 1);
}

static void validate_and_print(const char* tag, int* d_results, int num_tasks) {
    int* mismatch_count;
    CUDA_CHECK(cudaMallocManaged(&mismatch_count, sizeof(int)));
    *mismatch_count = 0;
    validate_double_results<<<(num_tasks + 255) / 256, 256>>>(d_results, num_tasks, mismatch_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (*mismatch_count != 0) printf("%s_Validate,FAIL,%d\n", tag, *mismatch_count);
    else printf("%s_Validate,PASS\n", tag);
    cudaFree(mismatch_count);
}

// 1. Baseline (Sync) - The "Worst Case"
void run_baseline_sync(int num_tasks, int num_iterations) {
    int* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, num_tasks * sizeof(int)));
    
    std::vector<double> times;
    times.reserve(num_iterations);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_tasks; i++) {
            baseline_task_kernel<<<1, 1>>>(d_results, i);
            cudaDeviceSynchronize(); 
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    // Stats
    double mean = 0;
    for (double t : times) mean += t;
    mean /= times.size();
    double throughput = (num_tasks * 1000.0) / mean;

    validate_and_print("Baseline_Sync", d_results, num_tasks);
    printf("Baseline_Sync,%d,%.3f,%.2f\n", num_tasks, mean, throughput);
    cudaFree(d_results);
}

// 2. Baseline (Batched) - The "Fair" CPU Benchmark
void run_baseline_batched(int num_tasks, int num_iterations) {
    int* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, num_tasks * sizeof(int)));
    
    std::vector<double> times;
    times.reserve(num_iterations);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        
        // Launch all tasks without intermediate sync
        // Note: For very large N, this might hit queue limits, but for <50k it should be fine.
        for (int i = 0; i < num_tasks; i++) {
            baseline_task_kernel<<<1, 1>>>(d_results, i);
        }
        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize(); // Sync ONCE at the end
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    // Stats
    double mean = 0;
    for (double t : times) mean += t;
    mean /= times.size();
    double throughput = (num_tasks * 1000.0) / mean;

    // Validate
    validate_and_print("Baseline_Batched", d_results, num_tasks);

    printf("Baseline_Batched,%d,%.3f,%.2f\n", num_tasks, mean, throughput);
    cudaFree(d_results);
}

// 3. Baseline (Bulk) - One kernel launch (upper bound if tasks are fusible)
void run_baseline_bulk(int num_tasks, int num_iterations) {
    int* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, num_tasks * sizeof(int)));

    std::vector<double> times;
    times.reserve(num_iterations);

    for (int iter = 0; iter < num_iterations; iter++) {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        int threads = 256;
        int blocks = (num_tasks + threads - 1) / threads;
        baseline_bulk_kernel<<<blocks, threads>>>(d_results, num_tasks);
        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    double mean = 0;
    for (double t : times) mean += t;
    mean /= times.size();
    double throughput = (num_tasks * 1000.0) / mean;

    validate_and_print("Baseline_Bulk", d_results, num_tasks);
    printf("Baseline_Bulk,%d,%.3f,%.2f\n", num_tasks, mean, throughput);
    cudaFree(d_results);
}

// 4. Baseline (CUDA Graphs) - Capture N tiny launches once, then replay
// This is a fairer "optimized CPU control plane" baseline when the task DAG is static.
void run_baseline_graphs(int num_tasks, int num_iterations) {
    int* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, num_tasks * sizeof(int)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Build graph once.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;

    // Capture the launch sequence.
    cudaError_t capStart = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (capStart != cudaSuccess) {
        printf("Baseline_Graphs,SKIP,capture_begin_failed,%s\n", cudaGetErrorString(capStart));
        cudaStreamDestroy(stream);
        cudaFree(d_results);
        return;
    }

    for (int i = 0; i < num_tasks; i++) {
        baseline_task_kernel<<<1, 1, 0, stream>>>(d_results, i);
    }
    cudaError_t capErr = cudaGetLastError();
    if (capErr != cudaSuccess) {
        // Abort capture cleanly.
        cudaStreamEndCapture(stream, &graph);
        printf("Baseline_Graphs,SKIP,capture_launch_failed,%s\n", cudaGetErrorString(capErr));
        if (graph) cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        cudaFree(d_results);
        return;
    }

    cudaError_t capEnd = cudaStreamEndCapture(stream, &graph);
    if (capEnd != cudaSuccess || graph == nullptr) {
        printf("Baseline_Graphs,SKIP,capture_end_failed,%s\n", cudaGetErrorString(capEnd));
        if (graph) cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        cudaFree(d_results);
        return;
    }

    cudaError_t instErr = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (instErr != cudaSuccess || exec == nullptr) {
        printf("Baseline_Graphs,SKIP,instantiate_failed,%s\n", cudaGetErrorString(instErr));
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        cudaFree(d_results);
        return;
    }

    std::vector<double> times;
    times.reserve(num_iterations);

    for (int iter = 0; iter < num_iterations; iter++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    double mean = 0;
    for (double t : times) mean += t;
    mean /= times.size();
    double throughput = (num_tasks * 1000.0) / mean;

    validate_and_print("Baseline_Graphs", d_results, num_tasks);
    printf("Baseline_Graphs,%d,%.3f,%.2f\n", num_tasks, mean, throughput);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_results);
}

// 5. GROSR - The GPU-Resident Runtime
void run_grosr_throughput(int num_tasks, int num_iterations) {
    int* d_results;
    CUDA_CHECK(cudaMallocManaged(&d_results, num_tasks * sizeof(int)));
    
    std::vector<double> times;
    times.reserve(num_iterations);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        TaskQueue q;
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
        
        grosr_runtime_kernel<<<1, 256>>>(runtime);
        CUDA_CHECK(cudaGetLastError());
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_tasks; i++) {
             while (__atomic_load_n(q.head, __ATOMIC_RELAXED) - __atomic_load_n(q.tail, __ATOMIC_RELAXED) >= q.capacity) {}
             SimpleTask task;
             task.task_id = i;
             task.data = i;
             push_task(&q, task);
        }
        
        while (*(volatile int*)q.tail < num_tasks) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
        
        *stop_flag = 1;
        cudaDeviceSynchronize();
        cleanup_task_queue(&q);
        cudaFree((int*)stop_flag);
    }
    
    double mean = 0;
    for (double t : times) mean += t;
    mean /= times.size();
    double throughput = (num_tasks * 1000.0) / mean;

    validate_and_print("GROSR", d_results, num_tasks);
    printf("GROSR_Throughput,%d,%.3f,%.2f\n", num_tasks, mean, throughput);
    cudaFree(d_results);
}

int main(int argc, char** argv) {
    int num_tasks = 10000;
    int num_iterations = 10;
    if (argc > 1) num_tasks = atoi(argv[1]);
    if (argc > 2) num_iterations = atoi(argv[2]);
    
    printf("Experiment B+: Fair Throughput Benchmark\n");
    printf("Method,Tasks,Mean_ms,Throughput_ops_sec\n");
    
    run_baseline_sync(num_tasks, num_iterations);
    run_baseline_batched(num_tasks, num_iterations);
    run_baseline_bulk(num_tasks, num_iterations);
    run_baseline_graphs(num_tasks, num_iterations);
    run_grosr_throughput(num_tasks, num_iterations);
    
    return 0;
}
