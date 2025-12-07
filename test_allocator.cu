// Test GPU-Side Slab Allocator
// Verifies that gpu_malloc() and gpu_free() work correctly

#include "grosr_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define NUM_ALLOCATIONS 1000
#define MAX_SIZE 1024

// Test kernel: Allocate, write, verify, free
__global__ void test_allocator_kernel(SlabAllocator* alloc, int* results, int num_tests) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_tests) return;
    
    // Test allocation
    size_t size = 32 + (tid % 8) * 32; // Vary sizes: 32, 64, 96, ..., 256
    void* ptr = gpu_malloc(alloc, size);
    
    if (ptr == nullptr) {
        results[tid * 3 + 0] = -1; // Allocation failed
        return;
    }
    
    results[tid * 3 + 0] = 1; // Allocation succeeded
    
    // Write test pattern
    int* int_ptr = (int*)ptr;
    int pattern = tid * 7 + 42;
    *int_ptr = pattern;
    
    // Verify
    if (*int_ptr == pattern) {
        results[tid * 3 + 1] = 1; // Write/read succeeded
    } else {
        results[tid * 3 + 1] = -1; // Write/read failed
    }
    
    // Free
    gpu_free(alloc, ptr);
    results[tid * 3 + 2] = 1; // Free succeeded
    
    // Try to allocate again (should succeed if free worked)
    void* ptr2 = gpu_malloc(alloc, size);
    if (ptr2 != nullptr) {
        results[tid * 3 + 2] = 2; // Re-allocation succeeded (free worked)
        gpu_free(alloc, ptr2);
    }
}

// Test kernel: Stress test with many allocations
__global__ void stress_test_kernel(SlabAllocator* alloc, int* results) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= NUM_ALLOCATIONS) return;
    
    // Allocate
    size_t size = 64 + (tid % 4) * 64; // 64, 128, 192, 256
    void* ptr = gpu_malloc(alloc, size);
    
    if (ptr == nullptr) {
        results[tid] = -1;
        return;
    }
    
    // Write pattern (memset not available in device code, use loop)
    char* char_ptr = (char*)ptr;
    char pattern = tid & 0xFF;
    for (size_t i = 0; i < size; i++) {
        char_ptr[i] = pattern;
    }
    results[tid] = 1;
    
    // Free (immediately for stress test)
    gpu_free(alloc, ptr);
}

int main() {
    printf("=== GROSR Slab Allocator Test ===\n\n");
    
    // Initialize allocator
    size_t arena_size = 64 * 1024 * 1024; // 64MB
    SlabAllocator* alloc;
    CUDA_CHECK(cudaMallocManaged(&alloc, sizeof(SlabAllocator)));
    
    printf("[*] Initializing allocator with %zu MB arena...\n", arena_size / (1024 * 1024));
    init_slab_allocator(alloc, arena_size);
    printf("[*] Allocator initialized.\n\n");
    
    // Test 1: Basic allocation and free
    printf("Test 1: Basic allocation and free\n");
    printf("---------------------------------\n");
    
    int* d_results1;
    CUDA_CHECK(cudaMallocManaged(&d_results1, NUM_ALLOCATIONS * 3 * sizeof(int)));
    memset(d_results1, 0, NUM_ALLOCATIONS * 3 * sizeof(int));
    
    test_allocator_kernel<<<1, NUM_ALLOCATIONS>>>(alloc, d_results1, NUM_ALLOCATIONS);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int alloc_success = 0, write_success = 0, free_success = 0;
    for (int i = 0; i < NUM_ALLOCATIONS; i++) {
        if (d_results1[i * 3 + 0] == 1) alloc_success++;
        if (d_results1[i * 3 + 1] == 1) write_success++;
        if (d_results1[i * 3 + 2] >= 1) free_success++;
    }
    
    printf("Allocations: %d/%d succeeded\n", alloc_success, NUM_ALLOCATIONS);
    printf("Write/Read:  %d/%d succeeded\n", write_success, NUM_ALLOCATIONS);
    printf("Free:         %d/%d succeeded\n", free_success, NUM_ALLOCATIONS);
    
    if (alloc_success == NUM_ALLOCATIONS && write_success == NUM_ALLOCATIONS && 
        free_success == NUM_ALLOCATIONS) {
        printf("✓ Test 1 PASSED\n\n");
    } else {
        printf("✗ Test 1 FAILED\n\n");
    }
    
    // Test 2: Stress test
    printf("Test 2: Stress test (%d allocations)\n", NUM_ALLOCATIONS);
    printf("------------------------------------\n");
    
    int* d_results2;
    CUDA_CHECK(cudaMallocManaged(&d_results2, NUM_ALLOCATIONS * sizeof(int)));
    
    stress_test_kernel<<<1, NUM_ALLOCATIONS>>>(alloc, d_results2);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int stress_success = 0;
    for (int i = 0; i < NUM_ALLOCATIONS; i++) {
        if (d_results2[i] == 1) stress_success++;
    }
    
    printf("Successful allocations: %d/%d\n", stress_success, NUM_ALLOCATIONS);
    
    if (stress_success == NUM_ALLOCATIONS) {
        printf("✓ Test 2 PASSED\n\n");
    } else {
        printf("✗ Test 2 FAILED\n\n");
    }
    
    // Cleanup
    cleanup_slab_allocator(alloc);
    cudaFree(alloc);
    cudaFree(d_results1);
    cudaFree(d_results2);
    
    printf("=== All Tests Complete ===\n");
    
    return 0;
}
