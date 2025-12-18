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
#include <cstring>

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
    
    // Note: "level" is the number of BFS iterations (frontier expansions), i.e., number of kernel launches.
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
        
        // Visited state machine:
        // 0 = unvisited, 2 = enqueued, 1 = processed.
        // If we dequeue a node that was marked "enqueued" (2), we MUST still process it.
        int old_visited = atomicExch(&result.visited[node], 1);
        if (old_visited == 1) continue; // Already processed
        
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
                while (atomicAdd_system(q->head, 0) - atomicAdd_system(q->tail, 0) >= q->capacity) {
                    __nanosleep(100);
                }
                
                int head = atomicAdd_system(q->head, 0);
                int slot = head % q->capacity;
                BFSTask* task_ptr = (BFSTask*)((char*)q->tasks + (slot * q->task_size));
                *task_ptr = neighbor_task;
                __threadfence_system();
                atomicAdd_system(q->head, 1);
            }
        }
        
        __threadfence_system();
    }
}

void run_grosr_bfs(Graph graph, int source, BFSResult result) {
    TaskQueue* q;
    CUDA_CHECK(cudaMallocManaged(&q, sizeof(TaskQueue)));
    // Ensure capacity is large enough to avoid producer/consumer deadlock in this single-thread runtime.
    // Each node is enqueued at most once (CAS 0->2), so pending tasks <= num_nodes.
    int capacity = graph.num_nodes + 64;
    if (capacity < 1024) capacity = 1024;
    init_task_queue(q, capacity, sizeof(BFSTask));
    
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
    // IMPORTANT: Do not call cudaDeviceSynchronize() here; the runtime kernel is persistent.
    
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
    
    // Note: last field is nodes processed (tasks executed), not BFS levels.
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

// Generate a simple chain graph: 0-1-2-...-(n-1) (undirected).
// This creates a high number of BFS levels (~n), amplifying CPU launch overhead
// in the baseline that launches one kernel per level.
Graph generate_chain_graph(int num_nodes) {
    Graph graph;
    graph.num_nodes = num_nodes;
    if (num_nodes <= 1) {
        graph.num_edges = 0;
        CUDA_CHECK(cudaMalloc(&graph.row_ptr, (num_nodes + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&graph.col_idx, 0));
        std::vector<int> h_row_ptr(num_nodes + 1, 0);
        CUDA_CHECK(cudaMemcpy(graph.row_ptr, h_row_ptr.data(),
                              (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
        return graph;
    }

    // Undirected chain: edges (i,i+1) and (i+1,i)
    int edge_count = 2 * (num_nodes - 1);
    graph.num_edges = edge_count;

    CUDA_CHECK(cudaMalloc(&graph.row_ptr, (num_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&graph.col_idx, edge_count * sizeof(int)));

    std::vector<int> h_row_ptr(num_nodes + 1);
    std::vector<int> h_col_idx(edge_count);

    int idx = 0;
    for (int i = 0; i < num_nodes; i++) {
        h_row_ptr[i] = idx;
        if (i > 0) h_col_idx[idx++] = i - 1;
        if (i + 1 < num_nodes) h_col_idx[idx++] = i + 1;
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

static void init_bfs_result(BFSResult* r, int num_nodes) {
    CUDA_CHECK(cudaMalloc(&r->distances, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&r->parents, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&r->visited, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(r->visited, 0, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(r->distances, 0xFF, num_nodes * sizeof(int))); // -1
    CUDA_CHECK(cudaMemset(r->parents, 0xFF, num_nodes * sizeof(int)));   // -1
}

void cleanup_bfs_result(BFSResult result, int num_nodes) {
    cudaFree(result.distances);
    cudaFree(result.parents);
    cudaFree(result.visited);
}

static bool validate_chain_bfs_host(int num_nodes, int source,
                                    const std::vector<int>& dist,
                                    const std::vector<int>& parent) {
    // Deterministic expected BFS tree for undirected chain with source=0:
    // dist[i] = i, parent[i] = i-1.
    if (source != 0) return false;
    if ((int)dist.size() != num_nodes || (int)parent.size() != num_nodes) return false;
    if (num_nodes <= 0) return true;
    if (dist[0] != 0) return false;
    if (parent[0] != -1) return false;
    for (int i = 1; i < num_nodes; i++) {
        if (dist[i] != i) return false;
        if (parent[i] != i - 1) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    int num_nodes = 10000;
    float edge_prob = 0.01f; // Sparse graph
    int source = 0;
    const char* graph_type = "random"; // "random" or "chain"
    
    if (argc > 1) num_nodes = atoi(argv[1]);
    if (argc > 2) edge_prob = atof(argv[2]);
    if (argc > 3) source = atoi(argv[3]);
    if (argc > 4) graph_type = argv[4];
    
    printf("Experiment C: Graph BFS Benchmark\n");
    if (strcmp(graph_type, "chain") == 0) {
        printf("Graph: %d nodes, type=chain\n", num_nodes);
    } else {
        printf("Graph: %d nodes, type=random, edge_prob=%.3f\n", num_nodes, edge_prob);
    }
    printf("Method,Nodes,Edges,Time_ms,Extra\n");
    
    // Generate graph
    Graph graph;
    if (strcmp(graph_type, "chain") == 0) {
        graph = generate_chain_graph(num_nodes);
    } else {
        graph = generate_random_graph(num_nodes, edge_prob);
    }
    printf("Generated graph with %d edges\n", graph.num_edges);
    
    // Allocate BFS results
    BFSResult baseline_result, grosr_result;
    init_bfs_result(&baseline_result, num_nodes);
    init_bfs_result(&grosr_result, num_nodes);
    
    // Run baseline
    run_baseline_bfs(graph, source, baseline_result);
    
    // Run GROSR
    run_grosr_bfs(graph, source, grosr_result);

    // Validation (only for chain graphs with source=0; distances/parents are deterministic).
    if (strcmp(graph_type, "chain") == 0 && source == 0) {
        std::vector<int> h_base_dist(num_nodes), h_base_parent(num_nodes);
        std::vector<int> h_grosr_dist(num_nodes), h_grosr_parent(num_nodes);

        CUDA_CHECK(cudaMemcpy(h_base_dist.data(), baseline_result.distances, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_base_parent.data(), baseline_result.parents, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_grosr_dist.data(), grosr_result.distances, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_grosr_parent.data(), grosr_result.parents, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));

        bool base_ok = validate_chain_bfs_host(num_nodes, source, h_base_dist, h_base_parent);
        bool grosr_ok = validate_chain_bfs_host(num_nodes, source, h_grosr_dist, h_grosr_parent);

        printf("Baseline_Validate,%s\n", base_ok ? "PASS" : "FAIL");
        printf("GROSR_Validate,%s\n", grosr_ok ? "PASS" : "FAIL");
    }
    
    // Cleanup
    cleanup_graph(graph);
    cleanup_bfs_result(baseline_result, num_nodes);
    cleanup_bfs_result(grosr_result, num_nodes);
    
    return 0;
}
