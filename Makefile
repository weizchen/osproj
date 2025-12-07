# Makefile for GROSR Project
# GPU-Resident Operating System Runtime

NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_70 -std=c++14
NVCC_FLAGS_DEBUG = -O0 -g -arch=sm_70 -std=c++14

# Detect GPU architecture (optional - can override)
# For A100 use sm_80, for H100 use sm_90
ARCH ?= sm_70

# Source files
GROSR_SOURCES = grosr_allocator.cu grosr_queue.cu grosr_runtime.cu
GROSR_OBJECTS = $(GROSR_SOURCES:.cu=.o)

# Executables (main experiments)
MAIN_EXECUTABLES = exp_a_pingpong exp_b_throughput exp_c_graph_bfs test_allocator

# Legacy executables (original demos, kept for reference)
LEGACY_EXECUTABLES = bench0 exp0 exp1

EXECUTABLES = $(MAIN_EXECUTABLES) $(LEGACY_EXECUTABLES)

.PHONY: all clean test

all: $(MAIN_EXECUTABLES) $(LEGACY_EXECUTABLES)

# Build only main experiments (recommended)
main: $(MAIN_EXECUTABLES)

# Build only legacy demos
legacy: $(LEGACY_EXECUTABLES)

# Update architecture flag
NVCC_FLAGS := $(subst sm_70,$(ARCH),$(NVCC_FLAGS))

# Build GROSR library objects
%.o: %.cu grosr_runtime.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Benchmark Experiment 0 (original)
bench0: benchmark_exp0.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Experiment 0: Microkernel (original)
exp0: exp0_microkernel.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Experiment 1: Syscalls (original)
exp1: exp1_syscalls.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Experiment A: Ping-Pong Latency
exp_a_pingpong: exp_a_pingpong.cu $(GROSR_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $< $(GROSR_OBJECTS) -o $@

# Experiment B: Throughput (with statistics)
exp_b_throughput: exp_b_throughput.cu $(GROSR_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $< $(GROSR_OBJECTS) -o $@

# Experiment C: Graph BFS Macrobenchmark
exp_c_graph_bfs: exp_c_graph_bfs.cu $(GROSR_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $< $(GROSR_OBJECTS) -o $@

# Test allocator (need separate compilation for device functions)
test_allocator: test_allocator.cu grosr_allocator.cu grosr_queue.o grosr_runtime.o
	$(NVCC) $(NVCC_FLAGS) -dc test_allocator.cu -o test_allocator.o
	$(NVCC) $(NVCC_FLAGS) -dc grosr_allocator.cu -o grosr_allocator_dc.o
	$(NVCC) $(NVCC_FLAGS) test_allocator.o grosr_allocator_dc.o grosr_queue.o grosr_runtime.o -o $@
	rm -f test_allocator.o grosr_allocator_dc.o

clean:
	rm -f $(EXECUTABLES) $(GROSR_OBJECTS) *.png *.csv *.o

test: test_allocator
	./test_allocator

# Run all benchmarks
benchmark: exp_a_pingpong exp_b_throughput
	@echo "Running Experiment A: Ping-Pong Latency"
	./exp_a_pingpong 1000
	@echo ""
	@echo "Running Experiment B: Throughput"
	./exp_b_throughput 10000 10

help:
	@echo "GROSR Build System"
	@echo ""
	@echo "Main Targets:"
	@echo "  main        - Build main experiments (recommended)"
	@echo "  all         - Build all executables (including legacy)"
	@echo "  legacy      - Build only legacy demo executables"
	@echo "  clean       - Remove all build artifacts"
	@echo "  test        - Run allocator tests"
	@echo "  benchmark   - Run main benchmarks (exp_a, exp_b)"
	@echo ""
	@echo "Individual Experiments:"
	@echo "  exp_a_pingpong   - Experiment A: Ping-Pong Latency"
	@echo "  exp_b_throughput - Experiment B: Throughput Benchmark"
	@echo "  exp_c_graph_bfs  - Experiment C: Graph BFS Macrobenchmark"
	@echo "  test_allocator   - Allocator unit tests"
	@echo ""
	@echo "Examples:"
	@echo "  make main              # Build main experiments"
	@echo "  make ARCH=sm_80 main   # Build for A100"
	@echo "  make clean             # Clean build"
	@echo "  make benchmark         # Run benchmarks"

