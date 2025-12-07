import subprocess
import sys
import matplotlib.pyplot as plt
import csv
import os

# Configuration
SOURCE_FILE = "benchmark_exp0.cu"
BINARY_NAME = "bench0"
NUM_TASKS = 5000  # Number of kernel launches to simulate

def compile_code():
    print(f"[*] Compiling {SOURCE_FILE}...")
    # Detect architecture (optional, defaulting to usually safe sm_70 for V100/T4/RTX20xx+)
    # For A100 use sm_80, for H100 use sm_90
    cmd = ["nvcc", "-O2", "-arch=sm_70", SOURCE_FILE, "-o", BINARY_NAME]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error compiling:")
        print(result.stderr)
        sys.exit(1)
    print("[*] Compilation successful.")

def run_benchmark():
    print(f"[*] Running benchmark with {NUM_TASKS} tasks...")
    cmd = [f"./{BINARY_NAME}", str(NUM_TASKS)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running benchmark:")
        print(result.stderr)
        sys.exit(1)
    return result.stdout.strip().split('\n')

def parse_and_plot(output_lines):
    methods = []
    throughputs = []
    latencies = []

    print("[*] Results:")
    print(f"{'Method':<15} | {'Time (ms)':<10} | {'Ops/Sec':<15}")
    print("-" * 45)

    for line in output_lines:
        if "," not in line: continue
        parts = line.split(',')
        method = parts[0]
        tasks = int(parts[1])
        time_ms = float(parts[2])
        ops_sec = float(parts[3])

        print(f"{method:<15} | {time_ms:<10.3f} | {ops_sec:<15.0f}")

        methods.append(method)
        throughputs.append(ops_sec)
        # Calculate average latency per task in microseconds
        latencies.append((time_ms * 1000) / tasks) 

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Throughput (Higher is better)
    bars = ax1.bar(methods, throughputs, color=['#e74c3c', '#2ecc71'])
    ax1.set_title('Kernel Dispatch Throughput')
    ax1.set_ylabel('Operations per Second')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')

    # Plot 2: Latency (Lower is better)
    bars2 = ax2.bar(methods, latencies, color=['#e74c3c', '#2ecc71'])
    ax2.set_title('Average Dispatch Latency')
    ax2.set_ylabel('Microseconds (μs) per Task')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} μs',
                ha='center', va='bottom')

    plt.suptitle(f'Experiment 0: Standard Launch vs GROSR Persistent Runtime\n(N={NUM_TASKS} tasks)', fontsize=14)
    plt.tight_layout()
    
    filename = 'exp0_results.png'
    plt.savefig(filename)
    print(f"\n[*] Plot saved to {filename}")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: {SOURCE_FILE} not found. Please save the C++ code first.")
        sys.exit(1)
        
    compile_code()
    output = run_benchmark()
    parse_and_plot(output)