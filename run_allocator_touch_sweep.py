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

    # We don’t try to parse the program’s stdout into structured fields because its
    # output format may evolve; instead we record the raw output for traceability.
    # For plotting, you can either:
    # - adjust exp_d_allocator_bench to print a strict CSV row, or
    # - add a small parser tailored to your current output.
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "num_threads",
                "iters_per_thread",
                "size_bytes",
                "mode",
                "outstanding",
                "touch_bytes",
                "cmd",
                "stdout",
            ]
        )
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
                w.writerow([nt, args.iters, args.size, args.mode, args.outstanding, tb, " ".join(cmd), out])
                print(f"[ok] threads={nt} touch={tb} -> {out_path}")

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


