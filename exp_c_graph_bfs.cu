// Experiment C: Graph BFS Macrobenchmark
// Demonstrates real-world benefit in dynamic, irregular workload
// Baseline: CPU checks queue size → launches kernel → GPU processes → CPU checks again
// GROSR: GPU threads add neighbors to queue dynamically → GPU scheduler processes autonomously

#include "grosr_runtime.h"
#include <chrono>
#include <thread>
#include <vector>
#include <queue>
#include <algorithm>
#include <random>

// Graph representation: Compressed Sparse Row (CSR) format
struct Graph {
    int num_nodes;
    int num_edges;
    int* row_ptr;      // Row pointer array (size: num_nodes + 1)
    int* col_idx;      // Column indices (size: num_edges)
    int* edge_weights; // Optional: edge weights
};

// BFS Task: represents a node to process
struct BFSTask {
    int node_id;
    int level;         // BFS level (distance from source)
    int parent;       // Parent node in BFS tree
};

// BFS Result
struct BFSResult {
    int* distances;   // Distance from source to each node
    int* parents;      // Parent in BFS tree
    int* visited;       // Visited flag
};

// Baseline: CPU-orchestrated BFS
__global__ void bfs_kernel_baseline(Graph graph, int* frontier, int frontier_size, 
                                     int* next_frontier, int* next_size,
                                     BFSResult result, int current_level) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= frontier_size) return;
    
    int node = frontier[tid];
    if (result.visited[node]) return;
    
    result.visited[node] = 1;
    result.distances[node] = current_level;
    
    // Process neighbors
    int start = graph.row_ptr[node];
    int end = graph.row_ptr[node + 1];
    
    for (int i = start; i < end; i++) {
        int neighbor = graph.col_idx[i];
        if (!result.visited[neighbor]) {
            // Add to next frontier (using atomic)
            int idx = atomicAdd(next_size, 1);
            if (idx < graph.num_nodes) {
                next_frontier[idx] = neighbor;
                result.parents[neighbor] = node;
            }
        }
    }
}

void run_baseline_bfs(Graph graph, int source, BFSResult result) {
    int* d_frontier;
    int* d_next_frontier;
    int* d_frontier_size;
    int* d_next_size;
    
    CUDA_CHECK(cudaMalloc(&d_frontier, graph.num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, graph.num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_size, sizeof(int)));
    
    // Initialize
    int h_frontier_size = 1;
    CUDA_CHECK(cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier_size, &h_frontier_size, sizeof(int), cudaMemcpyHostToDevice));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    int level = 0;
    while (h_frontier_size > 0) {
        // Launch kernel
        int threads = 256;
        int blocks = (h_frontier_size + threads - 1) / threads;
        bfs_kernel_baseline<<<blocks, threads>>>(graph, d_frontier, h_frontier_size,
                                                  d_next_frontier, d_next_size,
                                                  result, level);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Swap frontiers
        std::swap(d_frontier, d_next_frontier);
        CUDA_CHECK(cudaMemcpy(&h_frontier_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(d_next_size, 0, sizeof(int)));
        
        level++;
        if (level > graph.num_nodes) break; // Safety check
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Baseline_BFS,%d,%d,%.3f,%d\n", graph.num_nodes, graph.num_edges, ms, level);
    
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_frontier_size);
    cudaFree(d_next_size);
}

// GROSR: GPU-resident BFS
__global__ void grosr_bfs_runtime(TaskQueue* q, Graph graph, BFSResult result, 
                                   volatile int* stop_flag, int* tasks_processed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return; // Only scheduler thread
    
    while (*stop_flag == 0) {
        BFSTask task;
        if (!pop_task(q, &task, sizeof(BFSTask))) {
            __nanosleep(1000);
            continue;
        }
        
        // Process BFS task
        int node = task.node_id;
        
        // Check if already visited (with atomic to handle race conditions)
        int old_visited = atomicExch(&result.visited[node], 1);
        if (old_visited) continue; // Already processed
        
        result.distances[node] = task.level;
        result.parents[node] = task.parent;
        atomicAdd(tasks_processed, 1);
        
        // Process neighbors
        int start = graph.row_ptr[node];
        int end = graph.row_ptr[node + 1];
        
        for (int i = start; i < end; i++) {
            int neighbor = graph.col_idx[i];
            
            // Check if neighbor is unvisited
            int neighbor_visited = atomicAdd(&result.visited[neighbor], 0);
            if (neighbor_visited) continue;
            
            // Try to mark as "being processed" (optimistic)
            int claimed = atomicCAS(&result.visited[neighbor], 0, 2); // 2 = in queue
            if (claimed == 0) {
                // Add neighbor to queue
                BFSTask neighbor_task;
                neighbor_task.node_id = neighbor;
                neighbor_task.level = task.level + 1;
                neighbor_task.parent = node;
                
                // Push to queue (simple version - wait for space)
                while (*q->head - *q->tail >= q->capacity) {
                    __nanosleep(100);
                }
                
                int slot = (*q->head) % q->capacity;
                BFSTask* task_ptr = (BFSTask*)((char*)q->tasks + (slot * q->task_size));
                *task_ptr = neighbor_task;
                __threadfence_system();
                atomicAdd(q->head, 1);
            }
        }
        
        __threadfence_system();
    }
}

void run_grosr_bfs(Graph graph, int source, BFSResult result) {
    TaskQueue* q;
    CUDA_CHECK(cudaMallocManaged(&q, sizeof(TaskQueue)));
    init_task_queue(q, 16384, sizeof(BFSTask)); // Larger queue for BFS
    
    volatile int* stop_flag;
    CUDA_CHECK(cudaMallocManaged((int**)&stop_flag, sizeof(int)));
    *stop_flag = 0;
    
    int* d_tasks_processed;
    CUDA_CHECK(cudaMallocManaged(&d_tasks_processed, sizeof(int)));
    *d_tasks_processed = 0;
    
    // Launch persistent runtime
    grosr_bfs_runtime<<<1, 1>>>(q, graph, result, stop_flag, d_tasks_processed);
    CUDA_CHECK(cudaGetLastError());
    
    // Warmup
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    cudaDeviceSynchronize();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Push source node
    BFSTask source_task;
    source_task.node_id = source;
    source_task.level = 0;
    source_task.parent = -1;
    push_task(q, source_task);
    
    // Wait for completion (all nodes processed)
    int last_processed = 0;
    int stable_count = 0;
    while (*stop_flag == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        int current = *d_tasks_processed;
        
        if (current == last_processed) {
            stable_count++;
            // Check if queue is empty (host-side check)
            int head = *(volatile int*)q->head;
            int tail = *(volatile int*)q->tail;
            if (stable_count > 10 && head == tail) {
                break; // Queue empty and no progress
            }
        } else {
            stable_count = 0;
            last_processed = current;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("GROSR_BFS,%d,%d,%.3f,%d\n", graph.num_nodes, graph.num_edges, ms, *d_tasks_processed);
    
    // Cleanup
    *stop_flag = 1;
    cudaDeviceSynchronize();
    cleanup_task_queue(q);
    cudaFree(q);
    cudaFree((int*)stop_flag);
    cudaFree(d_tasks_processed);
}

// Generate a random graph (Erdos-Renyi model)
Graph generate_random_graph(int num_nodes, float edge_probability) {
    Graph graph;
    graph.num_nodes = num_nodes;
    
    std::vector<std::vector<int>> adj_list(num_nodes);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    // Generate edges
    int edge_count = 0;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = i + 1; j < num_nodes; j++) {
            if (dis(gen) < edge_probability) {
                adj_list[i].push_back(j);
                adj_list[j].push_back(i);
                edge_count += 2;
            }
        }
    }
    
    graph.num_edges = edge_count;
    
    // Convert to CSR format
    CUDA_CHECK(cudaMalloc(&graph.row_ptr, (num_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&graph.col_idx, edge_count * sizeof(int)));
    
    std::vector<int> h_row_ptr(num_nodes + 1);
    std::vector<int> h_col_idx(edge_count);
    
    int idx = 0;
    for (int i = 0; i < num_nodes; i++) {
        h_row_ptr[i] = idx;
        for (int neighbor : adj_list[i]) {
            h_col_idx[idx++] = neighbor;
        }
    }
    h_row_ptr[num_nodes] = idx;
    
    CUDA_CHECK(cudaMemcpy(graph.row_ptr, h_row_ptr.data(), 
                         (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(graph.col_idx, h_col_idx.data(),
                         edge_count * sizeof(int), cudaMemcpyHostToDevice));
    
    return graph;
}

void cleanup_graph(Graph graph) {
    cudaFree(graph.row_ptr);
    cudaFree(graph.col_idx);
}

void cleanup_bfs_result(BFSResult result, int num_nodes) {
    cudaFree(result.distances);
    cudaFree(result.parents);
    cudaFree(result.visited);
}

int main(int argc, char** argv) {
    int num_nodes = 10000;
    float edge_prob = 0.01f; // Sparse graph
    int source = 0;
    
    if (argc > 1) num_nodes = atoi(argv[1]);
    if (argc > 2) edge_prob = atof(argv[2]);
    if (argc > 3) source = atoi(argv[3]);
    
    printf("Experiment C: Graph BFS Benchmark\n");
    printf("Graph: %d nodes, edge_prob=%.3f\n", num_nodes, edge_prob);
    printf("Method,Nodes,Edges,Time_ms,NodesProcessed\n");
    
    // Generate graph
    Graph graph = generate_random_graph(num_nodes, edge_prob);
    printf("Generated graph with %d edges\n", graph.num_edges);
    
    // Allocate BFS results
    BFSResult baseline_result, grosr_result;
    CUDA_CHECK(cudaMalloc(&baseline_result.distances, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&baseline_result.parents, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&baseline_result.visited, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(baseline_result.visited, 0, num_nodes * sizeof(int)));
    
    CUDA_CHECK(cudaMalloc(&grosr_result.distances, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grosr_result.parents, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grosr_result.visited, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(grosr_result.visited, 0, num_nodes * sizeof(int)));
    
    // Run baseline
    run_baseline_bfs(graph, source, baseline_result);
    
    // Reset for GROSR
    CUDA_CHECK(cudaMemset(grosr_result.visited, 0, num_nodes * sizeof(int)));
    
    // Run GROSR
    run_grosr_bfs(graph, source, grosr_result);
    
    // Cleanup
    cleanup_graph(graph);
    cleanup_bfs_result(baseline_result, num_nodes);
    cleanup_bfs_result(grosr_result, num_nodes);
    
    return 0;
}
