// Compile with: nvcc -O2 -arch=sm_70 exp1_syscalls.cu -o exp1

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <thread>
#include <atomic>

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA Error: %s\n", cudaGetErrorString(err)); exit(1); } }

// --- 1. Syscall Data Structures ---

enum SyscallType { SC_READ = 1, SC_WRITE = 2 };

struct SyscallReq {
    volatile int status; // 0=FREE, 1=PENDING, 2=DONE
    int type;            // SC_READ
    int fd;              // File descriptor (on host)
    size_t size;         // Bytes to read
    char* dev_ptr;       // Where to put data on GPU
    int result;          // Return value (bytes read)
};

// --- 2. Device Functions (The "LibC" for GPU) ---

__device__ int find_free_slot(SyscallReq* table, int N) {
    for (int i = 0; i < N; ++i) {
        // Try to grab a slot: atomicCAS(ptr, expected, desired)
        if (atomicCAS((int*)&table[i].status, 0, 1) == 0) {
            return i;
        }
    }
    return -1; // No slots (should implement waiting/backoff here)
}

__device__ int gpu_read(SyscallReq* table, int N, int fd, char* buf, size_t size) {
    int slot = find_free_slot(table, N);
    if (slot < 0) return -1; // "ENOMEM"

    // Fill request
    table[slot].type = SC_READ;
    table[slot].fd = fd;
    table[slot].size = size;
    table[slot].dev_ptr = buf;
    
    // Commit request (CPU will now see status=1)
    __threadfence_system(); 

    // BLOCKING WAIT (The "Trap")
    // In a real OS, we would switch threads here. 
    // Here, we spin-wait (busy wait).
    while (atomicAdd((int*)&table[slot].status, 0) != 2) {
        __nanosleep(1000);
    }

    int res = table[slot].result;

    // Free the slot
    __threadfence_system();
    table[slot].status = 0; 
    
    return res;
}

__global__ void user_kernel(SyscallReq* sys_table, int N, int fd, char* my_buf) {
    // This kernel pretends to be a user app doing I/O
    int bytes = gpu_read(sys_table, N, fd, my_buf, 1024);
    
    if (bytes > 0) {
        printf("[GPU] Read %d bytes. First 4 chars: %c%c%c%c\n", 
               bytes, my_buf[0], my_buf[1], my_buf[2], my_buf[3]);
    } else {
        printf("[GPU] Read failed or empty.\n");
    }
}

// --- 3. CPU Proxy (The "Kernel Syscall Handler") ---

void os_proxy_thread(SyscallReq* table, int N, volatile bool* running) {
    while (*running) {
        bool worked = false;
        for (int i = 0; i < N; ++i) {
            // Check for PENDING (1) requests
            if (table[i].status == 1) {
                worked = true;
                if (table[i].type == SC_READ) {
                    // Perform the I/O on CPU
                    std::vector<char> host_buf(table[i].size);
                    ssize_t bytes = read(table[i].fd, host_buf.data(), table[i].size);
                    
                    // Copy data to GPU buffer
                    if (bytes > 0) {
                        cudaMemcpy(table[i].dev_ptr, host_buf.data(), bytes, cudaMemcpyHostToDevice);
                    }
                    
                    table[i].result = (int)bytes;
                }
                
                // Mark as DONE (2)
                __sync_synchronize(); // Memory barrier
                table[i].status = 2;
            }
        }
        if (!worked) std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

// --- 4. Main ---

int main() {
    // Setup file to read
    int fd = open("test.txt", O_RDONLY);
    if (fd < 0) {
        // create dummy file if not exists
        system("echo 'Hello from the filesystem!' > test.txt");
        fd = open("test.txt", O_RDONLY);
    }

    // Allocate Syscall Table
    int N = 32;
    SyscallReq* sys_table;
    CUDA_CHECK(cudaMallocManaged(&sys_table, N * sizeof(SyscallReq)));
    for(int i=0; i<N; ++i) sys_table[i].status = 0;

    // Allocate GPU buffer for the user kernel
    char* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, 1024));

    // Start OS Proxy Thread
    volatile bool running = true;
    std::thread proxy(os_proxy_thread, sys_table, N, &running);

    // Launch User Kernel
    printf("[CPU] Launching GPU kernel requesting I/O...\n");
    user_kernel<<<1, 1>>>(sys_table, N, fd, d_buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    running = false;
    proxy.join();
    close(fd);
    cudaFree(sys_table);
    cudaFree(d_buf);
    
    printf("[CPU] Done.\n");
    return 0;
}