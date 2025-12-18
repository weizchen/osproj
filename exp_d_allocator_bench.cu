// Experiment D: Allocator Microbenchmark
// Compares GROSR gpu_malloc/gpu_free vs CUDA device malloc/free.
//
// Usage:
//   ./exp_d_allocator_bench [num_threads] [iters_per_thread] [size_bytes] [mode] [outstanding] [touch_bytes]
//
// Notes:
// - This benchmark measures device-side dynamic allocation throughput.
// - Device malloc/free requires setting cudaLimitMallocHeapSize.
//
// Modes:
//   0 = churn: alloc+touch+free each iteration (microbenchmark / best-case for slab)
//   1 = outstanding: maintain a small pool of live allocations per thread, periodically free/replace
//   2 = mixed: like (1) but with mixed allocation sizes (size_bytes is treated as the "base" size)

#include "grosr_runtime.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

static __device__ __forceinline__ void touch_memory(void* p, size_t n, int seed) {
    // Write a small pattern to ensure the allocation is actually used.
    // Keep it light so timing mostly reflects allocator overhead.
    unsigned char* c = (unsigned char*)p;
    if (n >= 4) {
        // write 4 bytes
        c[0] = (unsigned char)(seed);
        c[1] = (unsigned char)(seed >> 8);
        c[2] = (unsigned char)(seed >> 16);
        c[3] = (unsigned char)(seed >> 24);
    } else if (n > 0) {
        c[0] = (unsigned char)(seed);
    }
}

static __device__ __forceinline__ void touch_memory_bytes(void* p, size_t n, int seed, int touch_bytes) {
    if (!p || n == 0) return;
    if (touch_bytes <= 0) return;
    size_t tb = (size_t)touch_bytes;
    if (tb > n) tb = n;
    if (tb == 0) return;

    unsigned char* c = (unsigned char*)p;
    unsigned int pattern = (unsigned int)seed;

    size_t i = 0;
    for (; i + 4 <= tb; i += 4) {
        c[i + 0] = (unsigned char)(pattern);
        c[i + 1] = (unsigned char)(pattern >> 8);
        c[i + 2] = (unsigned char)(pattern >> 16);
        c[i + 3] = (unsigned char)(pattern >> 24);
        pattern = pattern * 1664525u + 1013904223u;
    }
    for (; i < tb; i++) {
        c[i] = (unsigned char)(pattern);
        pattern = pattern * 1664525u + 1013904223u;
    }
}

__device__ __forceinline__ int alloc_churn_grosr(SlabAllocator* alloc, int iters, int size_bytes, int tid, int touch_bytes) {
    int local = 0;
    for (int i = 0; i < iters; i++) {
        void* p = gpu_malloc(alloc, (size_t)size_bytes);
        if (!p) {
            local -= 1;
            continue;
        }
        touch_memory_bytes(p, (size_t)size_bytes, tid ^ (i * 1315423911), touch_bytes);
        gpu_free(alloc, p);
        local += 1;
    }
    return local;
}

__device__ __forceinline__ int alloc_churn_dev(int iters, int size_bytes, int tid, int touch_bytes) {
    int local = 0;
    for (int i = 0; i < iters; i++) {
        void* p = malloc((size_t)size_bytes);
        if (!p) {
            local -= 1;
            continue;
        }
        touch_memory_bytes(p, (size_t)size_bytes, tid ^ (i * 1315423911), touch_bytes);
        free(p);
        local += 1;
    }
    return local;
}

__device__ __forceinline__ int alloc_outstanding_grosr(SlabAllocator* alloc,
                                                       int iters,
                                                       int size_bytes,
                                                       int mode,
                                                       int outstanding,
                                                       int tid,
                                                       int touch_bytes) {
    // Maintain a pool of live allocations per thread. Each iteration frees one slot and allocates a new one.
    // This approximates a workload with object lifetimes and steady memory pressure.
    if (outstanding <= 0) outstanding = 1;
    if (outstanding > 64) outstanding = 64; // keep stack usage bounded

    void* ptrs[64];
    for (int i = 0; i < outstanding; i++) {
        ptrs[i] = nullptr;
    }

    // Initialize pool
    for (int i = 0; i < outstanding; i++) {
        int s = size_bytes;
        if (mode == 2) {
            // Mixed sizes around the base: {base, 2*base, 4*base, 8*base} capped at 4096
            int mult = 1 << ((tid + i) & 3);
            s = size_bytes * mult;
            if (s > 4096) s = 4096;
        }
        void* p = gpu_malloc(alloc, (size_t)s);
        if (p) {
            ptrs[i] = p;
            touch_memory_bytes(p, (size_t)s, tid ^ (i * 7334147), touch_bytes);
        }
    }

    int local = 0;
    for (int i = 0; i < iters; i++) {
        int slot = (tid + i) % outstanding;
        if (ptrs[slot]) {
            gpu_free(alloc, ptrs[slot]);
            ptrs[slot] = nullptr;
        }

        int s = size_bytes;
        if (mode == 2) {
            int mult = 1 << ((tid + i) & 3);
            s = size_bytes * mult;
            if (s > 4096) s = 4096;
        }

        void* p = gpu_malloc(alloc, (size_t)s);
        if (!p) {
            local -= 1;
            continue;
        }
        touch_memory_bytes(p, (size_t)s, tid ^ (i * 2654435761U), touch_bytes);
        ptrs[slot] = p;
        local += 1;
    }

    // Drain
    for (int i = 0; i < outstanding; i++) {
        if (ptrs[i]) gpu_free(alloc, ptrs[i]);
    }
    return local;
}

__device__ __forceinline__ int alloc_outstanding_dev(int iters,
                                                     int size_bytes,
                                                     int mode,
                                                     int outstanding,
                                                     int tid,
                                                     int touch_bytes) {
    if (outstanding <= 0) outstanding = 1;
    if (outstanding > 64) outstanding = 64;

    void* ptrs[64];
    for (int i = 0; i < outstanding; i++) ptrs[i] = nullptr;

    for (int i = 0; i < outstanding; i++) {
        int s = size_bytes;
        if (mode == 2) {
            int mult = 1 << ((tid + i) & 3);
            s = size_bytes * mult;
            if (s > 4096) s = 4096;
        }
        void* p = malloc((size_t)s);
        if (p) {
            ptrs[i] = p;
            touch_memory_bytes(p, (size_t)s, tid ^ (i * 7334147), touch_bytes);
        }
    }

    int local = 0;
    for (int i = 0; i < iters; i++) {
        int slot = (tid + i) % outstanding;
        if (ptrs[slot]) {
            free(ptrs[slot]);
            ptrs[slot] = nullptr;
        }

        int s = size_bytes;
        if (mode == 2) {
            int mult = 1 << ((tid + i) & 3);
            s = size_bytes * mult;
            if (s > 4096) s = 4096;
        }
        void* p = malloc((size_t)s);
        if (!p) {
            local -= 1;
            continue;
        }
        touch_memory_bytes(p, (size_t)s, tid ^ (i * 2654435761U), touch_bytes);
        ptrs[slot] = p;
        local += 1;
    }

    for (int i = 0; i < outstanding; i++) {
        if (ptrs[i]) free(ptrs[i]);
    }
    return local;
}

__global__ void bench_grosr_alloc(SlabAllocator* alloc, int iters, int size_bytes, int mode, int outstanding, int touch_bytes, int* out_sum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local = 0;
    if (mode == 0) local = alloc_churn_grosr(alloc, iters, size_bytes, tid, touch_bytes);
    else local = alloc_outstanding_grosr(alloc, iters, size_bytes, mode, outstanding, tid, touch_bytes);
    out_sum[tid] = local;
}

__global__ void bench_device_malloc(int iters, int size_bytes, int mode, int outstanding, int touch_bytes, int* out_sum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local = 0;
    if (mode == 0) local = alloc_churn_dev(iters, size_bytes, tid, touch_bytes);
    else local = alloc_outstanding_dev(iters, size_bytes, mode, outstanding, tid, touch_bytes);
    out_sum[tid] = local;
}

static void run_one(const char* name,
                    void (*launch)(int blocks, int threads, int iters, int size_bytes, int mode, int outstanding, int touch_bytes, int* out_sum, SlabAllocator* alloc),
                    int blocks, int threads, int iters, int size_bytes, int mode, int outstanding, int touch_bytes, int* out_sum, SlabAllocator* alloc,
                    float* out_ms,
                    long long* out_success,
                    long long* out_fail) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemset(out_sum, 0, blocks * threads * sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    launch(blocks, threads, iters, size_bytes, mode, outstanding, touch_bytes, out_sum, alloc);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    *out_ms = ms;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Compute success/fail counts from out_sum:
    // Each iteration is either success (+1) or fail (-1), so:
    //   local = success - fail
    //   iters = success + fail
    // => success = (local + iters)/2, fail = iters - success
    long long total_success = 0;
    long long total_fail = 0;
    int total_threads = blocks * threads;
    for (int i = 0; i < total_threads; i++) {
        int local = out_sum[i];
        int succ = (local + iters) / 2;
        int fail = iters - succ;
        total_success += succ;
        total_fail += fail;
    }
    *out_success = total_success;
    *out_fail = total_fail;

    // Basic sanity: successes + fails == total_ops.
    long long total_ops = (long long)total_threads * (long long)iters;
    if (total_success + total_fail != total_ops) {
        printf("%s_Warn,success_plus_fail_mismatch,%lld,%lld\n", name, total_ops, total_success + total_fail);
    }
}

static void launch_grosr(int blocks, int threads, int iters, int size_bytes, int mode, int outstanding, int touch_bytes, int* out_sum, SlabAllocator* alloc) {
    bench_grosr_alloc<<<blocks, threads>>>(alloc, iters, size_bytes, mode, outstanding, touch_bytes, out_sum);
    CUDA_CHECK(cudaGetLastError());
}

static void launch_dev_malloc(int blocks, int threads, int iters, int size_bytes, int mode, int outstanding, int touch_bytes, int* out_sum, SlabAllocator* /*alloc*/) {
    bench_device_malloc<<<blocks, threads>>>(iters, size_bytes, mode, outstanding, touch_bytes, out_sum);
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
    int num_threads = 65536;     // total threads
    int iters = 100;             // per thread
    int size_bytes = 64;         // allocation size
    int mode = 0;                // see Modes above
    int outstanding = 16;        // live allocations per thread for modes 1/2
    int touch_bytes = 4;         // bytes to touch per allocation (0 = no touch, <=size_bytes recommended)

    if (argc > 1) num_threads = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);
    if (argc > 3) size_bytes = atoi(argv[3]);
    if (argc > 4) mode = atoi(argv[4]);
    if (argc > 5) outstanding = atoi(argv[5]);
    if (argc > 6) touch_bytes = atoi(argv[6]);

    if (num_threads <= 0) num_threads = 1;
    if (iters <= 0) iters = 1;
    if (size_bytes <= 0) size_bytes = 1;
    if (mode < 0) mode = 0;
    if (mode > 2) mode = 2;
    if (touch_bytes < 0) touch_bytes = 0;

    int threads = 256;
    int blocks = (num_threads + threads - 1) / threads;

    // Device malloc/free needs a heap. Make it large enough for concurrency.
    // Worst case outstanding allocations is roughly num_threads if malloc/free is slow.
    // We set something generous but not insane.
    // For modes with outstanding allocations, provision heap proportional to outstanding.
    int eff_out = (mode == 0) ? 2 : (outstanding + 2);
    size_t heap_bytes = (size_t)num_threads * (size_t)size_bytes * (size_t)eff_out;
    if (heap_bytes < (size_t)64 * 1024 * 1024) heap_bytes = (size_t)64 * 1024 * 1024;
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_bytes));

    // GROSR allocator arena (managed memory)
    size_t arena_size = (size_t)128 * 1024 * 1024; // 128MB default
    SlabAllocator* alloc;
    CUDA_CHECK(cudaMallocManaged(&alloc, sizeof(SlabAllocator)));
    init_slab_allocator(alloc, arena_size);

    int* out_sum;
    CUDA_CHECK(cudaMallocManaged(&out_sum, blocks * threads * sizeof(int)));

    float ms_grosr = 0.0f, ms_dev = 0.0f;
    long long grosr_success = 0, grosr_fail = 0;
    long long dev_success = 0, dev_fail = 0;
    run_one("GROSR", launch_grosr, blocks, threads, iters, size_bytes, mode, outstanding, touch_bytes, out_sum, alloc, &ms_grosr, &grosr_success, &grosr_fail);
    run_one("DeviceMalloc", launch_dev_malloc, blocks, threads, iters, size_bytes, mode, outstanding, touch_bytes, out_sum, nullptr, &ms_dev, &dev_success, &dev_fail);

    double grosr_ops_sec = (double)grosr_success / (ms_grosr / 1000.0);
    double dev_ops_sec = (double)dev_success / (ms_dev / 1000.0);

    printf("Experiment D: Allocator Microbenchmark\n");
    printf("ThreadsTotal=%d, ItersPerThread=%d, SizeBytes=%d, Mode=%d, Outstanding=%d, TouchBytes=%d\n",
           blocks * threads, iters, size_bytes, mode, outstanding, touch_bytes);
    printf("Method,Threads,Iters,SizeBytes,Mode,Outstanding,TouchBytes,Time_ms,AllocsPerSec,SuccessRate\n");
    double total_ops = (double)blocks * (double)threads * (double)iters;
    double grosr_sr = (total_ops > 0) ? ((double)grosr_success / total_ops) : 0.0;
    double dev_sr = (total_ops > 0) ? ((double)dev_success / total_ops) : 0.0;
    printf("GROSR_Alloc,%d,%d,%d,%d,%d,%d,%.3f,%.0f,%.4f\n",
           blocks * threads, iters, size_bytes, mode, outstanding, touch_bytes, ms_grosr, grosr_ops_sec, grosr_sr);
    printf("Device_Malloc,%d,%d,%d,%d,%d,%d,%.3f,%.0f,%.4f\n",
           blocks * threads, iters, size_bytes, mode, outstanding, touch_bytes, ms_dev, dev_ops_sec, dev_sr);

    cleanup_slab_allocator(alloc);
    cudaFree(alloc);
    cudaFree(out_sum);
    return 0;
}


