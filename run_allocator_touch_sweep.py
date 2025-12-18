import subprocess
import sys


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout


def extract_lines(out: str) -> list[str]:
    keep = []
    for line in out.splitlines():
        if line.startswith("ThreadsTotal=") or line.startswith("Method,") or line.startswith("GROSR_Alloc,") or line.startswith("Device_Malloc,"):
            keep.append(line)
    return keep


def main() -> int:
    # Defaults chosen to run quickly but still show trends.
    exe = "./exp_d_allocator_bench"
    threads_total = [2048, 8192, 16384]
    iters = 200
    size_bytes = 256
    mode = 1
    outstanding = 16
    touch_bytes_list = [0, 4, 64, 256]

    print("# Allocator touch sweep")
    print(f"# iters={iters} size_bytes={size_bytes} mode={mode} outstanding={outstanding}")
    for t in threads_total:
        for tb in touch_bytes_list:
            print(f"\n--- threads={t} touch_bytes={tb}")
            out = run([exe, str(t), str(iters), str(size_bytes), str(mode), str(outstanding), str(tb)])
            for line in extract_lines(out):
                print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())


