// GROSR GPU-Side Slab Allocator Implementation
// Implements gpu_malloc() and gpu_free() entirely on GPU
// Simplified bitmap-based allocator

#include "grosr_runtime.h"
#include <cstdlib>
#include <cstring>

// Size classes: 32B, 64B, 128B, 256B, 512B, 1KB, 2KB, 4KB
// Use constexpr for device code compatibility
__device__ __host__ __forceinline__ int get_slab_size(int class_idx) {
    const int sizes[NUM_SLAB_CLASSES] = {
        SLAB_SIZE_32B, SLAB_SIZE_64B, SLAB_SIZE_128B, SLAB_SIZE_256B,
        SLAB_SIZE_512B, SLAB_SIZE_1KB, SLAB_SIZE_2KB, SLAB_SIZE_4KB
    };
    return sizes[class_idx];
}

// Find size class for a given size
__device__ __host__ int find_size_class(size_t size) {
    if (size <= SLAB_SIZE_32B) return 0;
    if (size <= SLAB_SIZE_64B) return 1;
    if (size <= SLAB_SIZE_128B) return 2;
    if (size <= SLAB_SIZE_256B) return 3;
    if (size <= SLAB_SIZE_512B) return 4;
    if (size <= SLAB_SIZE_1KB) return 5;
    if (size <= SLAB_SIZE_2KB) return 6;
    if (size <= SLAB_SIZE_4KB) return 7;
    return -1; // Too large
}

// Calculate number of blocks per slab for a given size class
__device__ __host__ int blocks_per_slab(int size_class) {
    // Use 4KB slabs, so blocks_per_slab = 4096 / block_size
    return 4096 / get_slab_size(size_class);
}

// Initialize slab allocator (CPU-side)
// Simplified: Each size class gets equal portion of arena, divided into 4KB slabs
void init_slab_allocator(SlabAllocator* alloc, size_t arena_size) {
    alloc->arena_size = arena_size;
    
    // Allocate arena in unified memory
    CUDA_CHECK(cudaMallocManaged(&alloc->arena, arena_size));
    memset(alloc->arena, 0, arena_size);
    
    // Divide arena into 4KB slabs
    const int SLAB_SIZE = 4096;
    int total_slabs = (int)(arena_size / SLAB_SIZE);
    if (total_slabs <= 0) {
        // Degenerate arena; allocator will just fail all allocations.
        alloc->slab_headers = nullptr;
        for (int i = 0; i < NUM_SLAB_CLASSES; i++) {
            alloc->num_slabs_per_class[i] = 0;
            alloc->slab_sizes[i] = get_slab_size(i);
            alloc->free_lists[i] = nullptr;
            alloc->free_list_tops[i] = nullptr;
        }
        return;
    }
    
    // Allocate slab headers (one per slab)
    CUDA_CHECK(cudaMallocManaged(&alloc->slab_headers, total_slabs * sizeof(SlabHeader)));
    
    // Initialize size classes and slab headers
    int slab_idx = 0;
    char* arena_ptr = alloc->arena;
    
    // Robust per-class slab distribution (sum == total_slabs)
    // This avoids overrunning slab_headers when total_slabs < NUM_SLAB_CLASSES.
    for (int class_idx = 0; class_idx < NUM_SLAB_CLASSES; class_idx++) {
        int block_size = get_slab_size(class_idx);
        alloc->slab_sizes[class_idx] = block_size;

        int slabs_this_class = total_slabs / NUM_SLAB_CLASSES;
        int remainder = total_slabs % NUM_SLAB_CLASSES;
        if (class_idx < remainder) slabs_this_class++;
        alloc->num_slabs_per_class[class_idx] = slabs_this_class;

        // Free list fields are currently unused by gpu_malloc/gpu_free.
        alloc->free_lists[class_idx] = nullptr;
        alloc->free_list_tops[class_idx] = nullptr;

        int blocks_per = blocks_per_slab(class_idx);

        for (int s = 0; s < slabs_this_class; s++) {
            SlabHeader* header = &alloc->slab_headers[slab_idx];
            header->block_size = block_size;
            header->num_blocks = blocks_per;

            // Initialize multi-word bitmap: bit=1 means free.
            // Supports up to 128 blocks per slab (32B class).
            for (int w = 0; w < 4; w++) {
                int start_bit = w * 32;
                int remaining = blocks_per - start_bit;
                if (remaining <= 0) {
                    header->free_mask[w] = 0u;
                } else if (remaining >= 32) {
                    header->free_mask[w] = 0xFFFFFFFFu;
                } else {
                    header->free_mask[w] = (1u << remaining) - 1u;
                }
            }

            slab_idx++;
            arena_ptr += SLAB_SIZE;
        }
    }
}

// Cleanup slab allocator
void cleanup_slab_allocator(SlabAllocator* alloc) {
    if (alloc->arena) cudaFree(alloc->arena);
    if (alloc->slab_headers) cudaFree(alloc->slab_headers);
    for (int i = 0; i < NUM_SLAB_CLASSES; i++) {
        if (alloc->free_lists[i]) cudaFree(alloc->free_lists[i]);
        if (alloc->free_list_tops[i]) cudaFree(alloc->free_list_tops[i]);
    }
}

// GPU-side allocation
__device__ void* gpu_malloc(SlabAllocator* alloc, size_t size) {
    int class_idx = find_size_class(size);
    if (class_idx < 0) return nullptr; // Too large
    
    int num_slabs = alloc->num_slabs_per_class[class_idx];
    int block_size = alloc->slab_sizes[class_idx];
    
    // Improved search strategy: start at random offset to reduce contention
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start_offset = tid % num_slabs;

    // Search through slabs for this class to find a free block
    for (int i = 0; i < num_slabs; i++) {
        int attempt = (start_offset + i) % num_slabs;

        // Calculate slab index for this class
        int base_slab = 0;
        for (int j = 0; j < class_idx; j++) {
            base_slab += alloc->num_slabs_per_class[j];
        }
        int slab_idx = base_slab + attempt;
        
        SlabHeader* header = &alloc->slab_headers[slab_idx];

        // Scan bitmap words for a free block
        for (int w = 0; w < 4; w++) {
            unsigned int mask = header->free_mask[w];
            if (mask == 0u) continue;

            int bit = __ffs((int)mask) - 1; // 0-based bit index within this word
            if (bit < 0) continue;

            int block_idx = w * 32 + bit;
            if (block_idx < 0 || block_idx >= header->num_blocks) continue;

            unsigned int claim_mask = 1u << bit;
            unsigned int old_mask = atomicAnd(&header->free_mask[w], ~claim_mask);
            if (old_mask & claim_mask) {
                char* slab_start = alloc->arena + (slab_idx * 4096);
                return slab_start + (block_idx * block_size);
            }
        }
    }
    
    return nullptr; // No free blocks
}

// GPU-side free
__device__ void gpu_free(SlabAllocator* alloc, void* ptr) {
    if (!ptr) return;
    
    // Calculate which slab this pointer belongs to
    size_t offset = (char*)ptr - alloc->arena;
    
    if (offset >= alloc->arena_size) return; // Invalid pointer
    
    int slab_idx = offset / 4096;
    if (slab_idx < 0) return;
    
    SlabHeader* header = &alloc->slab_headers[slab_idx];
    int block_size = header->block_size;
    
    // Calculate block index within slab
    int offset_in_slab = offset % 4096;
    int block_idx = offset_in_slab / block_size;
    
    if (block_idx < 0 || block_idx >= header->num_blocks) return;
    
    // Mark block as free atomically (multi-word bitmap)
    int word = block_idx / 32;
    int bit = block_idx % 32;
    if (word < 0 || word >= 4) return;
    unsigned int mask = 1u << bit;
    atomicOr(&header->free_mask[word], mask);
}
