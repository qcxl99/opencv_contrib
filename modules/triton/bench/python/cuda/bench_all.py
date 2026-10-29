"""Full regression entry point: runs every bench_correctness_*.py and
bench_performance_*.py in this directory as subprocesses, in order, and
reports a single pass/fail plus each script's own output.

Replaces bench_full_resize.py (correctness+performance+batch, modes
bundled together in one script) and bench_real_protocol.py (performance
only, qualification+performance deduped, all modes in one pass) -- both
retired by this split. This is the merged "run everything" entry those
two were heading towards from different directions: bench_full_resize.py
by mode-bundling, bench_real_protocol.py by fixing the measurement
protocol. Neither replaced the other; this does both -- one entry point,
the real benchmark_case() protocol, each mode/interface's own script
called rather than re-implemented here.

Run: LD_LIBRARY_PATH=/opt/ocv-cuda/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib \
     python3 bench/bench_all.py [--correctness-only | --performance-only]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

CORRECTNESS_SCRIPTS = [
    "bench_correctness_linear.py",
    "bench_correctness_nearest.py",
    "bench_correctness_area.py",
    "bench_correctness_cubic.py",
    "bench_correctness_batch.py",
]
PERFORMANCE_SCRIPTS = [
    "bench_performance_linear.py",
    "bench_performance_nearest.py",
    "bench_performance_area.py",
    "bench_performance_cubic.py",
    "bench_performance_batch.py",
]


def run_one(script: str) -> bool:
    print(f"\n{'#' * 78}\n# {script}\n{'#' * 78}")
    r = subprocess.run([sys.executable, str(HERE / script)], cwd=str(HERE))
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--correctness-only", action="store_true")
    ap.add_argument("--performance-only", action="store_true")
    args = ap.parse_args()

    scripts = []
    if not args.performance_only:
        scripts += CORRECTNESS_SCRIPTS
    if not args.correctness_only:
        scripts += PERFORMANCE_SCRIPTS

    results = {s: run_one(s) for s in scripts}

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    all_ok = True
    for s, ok in results.items():
        # Performance scripts have no pass/fail gate (per-case verdicts
        # are printed, not aggregated into one boolean) -- a nonzero exit
        # there means the script itself errored, not a performance
        # regression; correctness scripts' exit code IS the gate.
        tag = "PASS" if ok else ("FAIL" if s in CORRECTNESS_SCRIPTS else "ERRORED")
        print(f"  {tag:8s} {s}")
        if s in CORRECTNESS_SCRIPTS:
            all_ok &= ok

    print(f"\nOVERALL CORRECTNESS: {'PASS' if all_ok else 'FAIL'}")
    print("(performance scripts report per-case verdicts in their own output "
          "above; MEASUREMENT_INVALID cases are flagged there, not silently "
          "scored into this summary)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
