#!/usr/bin/env python3
"""
Helper script for Exp D (allocator microbenchmark).

Runs ./exp_d_allocator_bench across a sweep of touch_bytes (and optionally thread counts)
and writes a simple CSV so you can paste results into the report.

This script is optional: the project does not depend on it for building/running.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime


def run_cmd(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")
    return p.stdout


def parse_exp_d_stdout(out: str) -> dict[str, dict[str, str]]:
    """
    Parse exp_d_allocator_bench stdout.

    The program prints CSV rows like:
      Method,Threads,...,Time_ms,AllocsPerSec,SuccessRate
      GROSR_Alloc,...
      Device_Malloc,...
      ThreadPool_Suballoc,...

    Returns:
      dict: method_name -> {column_name -> value_as_string}
    """
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    header: list[str] | None = None
    rows: dict[str, dict[str, str]] = {}
    for ln in lines:
        if ln.startswith("Method,"):
            header = [h.strip() for h in ln.split(",")]
            continue
        if header is None:
            continue
        # Data rows begin with a method name and have same column count as header.
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(header):
            continue
        method = parts[0]
        rows[method] = {header[i]: parts[i] for i in range(len(header))}
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="./exp_d_allocator_bench", help="path to exp_d_allocator_bench binary")
    ap.add_argument("--threads", default="2048", help="comma-separated num_threads values (e.g., 2048,4096,16384)")
    ap.add_argument("--iters", type=int, default=200, help="iters_per_thread")
    ap.add_argument("--size", type=int, default=256, help="size_bytes")
    ap.add_argument("--mode", type=int, default=1, help="mode (0 churn, 1 outstanding, 2 mixed)")
    ap.add_argument("--outstanding", type=int, default=16, help="outstanding (mode 1/2)")
    ap.add_argument(
        "--touch",
        default="4,64,256,1024",
        help="comma-separated touch_bytes values (e.g., 4,64,256)",
    )
    ap.add_argument("--out", default="", help="output CSV path (default: sweep_<timestamp>.csv)")
    ap.add_argument(
        "--raw",
        action="store_true",
        help="also include full program stdout in the CSV (useful for debugging)",
    )
    args = ap.parse_args()

    bin_path = args.bin
    if not os.path.exists(bin_path):
        print(f"ERROR: binary not found: {bin_path}", file=sys.stderr)
        print("Build it with: make ARCH=sm_XX exp_d_allocator_bench", file=sys.stderr)
        return 2

    threads_list = [int(x) for x in args.threads.split(",") if x.strip()]
    touch_list = [int(x) for x in args.touch.split(",") if x.strip()]

    out_path = args.out.strip()
    if not out_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"sweep_alloc_touch_{ts}.csv"

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        cols = [
            "num_threads",
            "iters_per_thread",
            "size_bytes",
            "mode",
            "outstanding",
            "touch_bytes",
            "grosr_allocs_per_sec",
            "device_allocs_per_sec",
            "pool_allocs_per_sec",
            "grosr_success_rate",
            "device_success_rate",
            "pool_success_rate",
            "grosr_time_ms",
            "device_time_ms",
            "pool_time_ms",
            "cmd",
        ]
        if args.raw:
            cols.append("stdout")
        w.writerow(cols)

        for nt in threads_list:
            for tb in touch_list:
                cmd = [
                    bin_path,
                    str(nt),
                    str(args.iters),
                    str(args.size),
                    str(args.mode),
                    str(args.outstanding),
                    str(tb),
                ]
                out = run_cmd(cmd).strip()
                rows = parse_exp_d_stdout(out)

                def get(method: str, key: str) -> str:
                    return rows.get(method, {}).get(key, "")

                grosr_ops = get("GROSR_Alloc", "AllocsPerSec")
                dev_ops = get("Device_Malloc", "AllocsPerSec")
                pool_ops = get("ThreadPool_Suballoc", "AllocsPerSec")
                grosr_sr = get("GROSR_Alloc", "SuccessRate")
                dev_sr = get("Device_Malloc", "SuccessRate")
                pool_sr = get("ThreadPool_Suballoc", "SuccessRate")
                grosr_ms = get("GROSR_Alloc", "Time_ms")
                dev_ms = get("Device_Malloc", "Time_ms")
                pool_ms = get("ThreadPool_Suballoc", "Time_ms")

                row = [
                    nt,
                    args.iters,
                    args.size,
                    args.mode,
                    args.outstanding,
                    tb,
                    grosr_ops,
                    dev_ops,
                    pool_ops,
                    grosr_sr,
                    dev_sr,
                    pool_sr,
                    grosr_ms,
                    dev_ms,
                    pool_ms,
                    " ".join(cmd),
                ]
                if args.raw:
                    row.append(out)
                w.writerow(row)
                print(f"[ok] threads={nt} touch={tb} -> {out_path}")

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


