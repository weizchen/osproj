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
    
    // Calculate slabs per class (simplified: equal distribution)
    const int SLAB_SIZE = 4096;
    int total_slabs = arena_size / SLAB_SIZE;
    int slabs_per_class = total_slabs / NUM_SLAB_CLASSES;
    if (slabs_per_class < 1) slabs_per_class = 1;
    
    // Allocate slab headers (one per slab)
    CUDA_CHECK(cudaMallocManaged(&alloc->slab_headers, total_slabs * sizeof(SlabHeader)));
    
    // Initialize size classes and slab headers
    int slab_idx = 0;
    char* arena_ptr = alloc->arena;
    
    for (int class_idx = 0; class_idx < NUM_SLAB_CLASSES; class_idx++) {
        int block_size = get_slab_size(class_idx);
        alloc->slab_sizes[class_idx] = block_size;
        alloc->num_slabs_per_class[class_idx] = slabs_per_class;
        int blocks_per = blocks_per_slab(class_idx);
        
        // Allocate free list for this class
        CUDA_CHECK(cudaMallocManaged(&alloc->free_lists[class_idx], slabs_per_class * sizeof(int)));
        CUDA_CHECK(cudaMallocManaged(&alloc->free_list_tops[class_idx], sizeof(int)));
        *alloc->free_list_tops[class_idx] = slabs_per_class;
        
        // Initialize each slab in this class
        for (int s = 0; s < slabs_per_class; s++) {
            SlabHeader* header = &alloc->slab_headers[slab_idx];
            header->block_size = block_size;
            header->num_blocks = blocks_per;
            // Initialize free mask: all blocks free (all bits set to 1)
            // For up to 32 blocks, use single int; for more, would need array
            if (blocks_per < 32) { // Fix: blocks_per=32 should use full mask (UB for shift 32)
                header->free_mask = (1U << blocks_per) - 1;
            } else {
                header->free_mask = 0xFFFFFFFF; // Max for 32-bit
            }
            header->next_free = 0;
            
            // Add to free list
            alloc->free_lists[class_idx][s] = slab_idx;
            
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
        
        if (header->free_mask == 0) continue; // Slab is full
        
        // Find first free block using bit scan forward
        unsigned int mask = header->free_mask;
        int block_idx = __ffs((int)mask) - 1; // ffs returns 1-based index
        
        if (block_idx < 0 || block_idx >= header->num_blocks) continue;
        
        // Try to claim the block atomically
        unsigned int claim_mask = 1U << block_idx;
        unsigned int old_mask = atomicAnd(&header->free_mask, ~claim_mask);
        
        // Check if we successfully claimed it
        if (old_mask & claim_mask) {
            // Calculate pointer to block
            // Slabs are stored sequentially: slab_idx * 4096 gives offset
            char* slab_start = alloc->arena + (slab_idx * 4096);
            void* block_ptr = slab_start + (block_idx * block_size);
            return block_ptr;
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
    
    // Mark block as free atomically
    unsigned int mask = 1U << block_idx;
    atomicOr(&header->free_mask, mask);
}
