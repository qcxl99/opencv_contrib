"""INTER_CUBIC performance vs cv::cuda::resize.

Uses cases.yaml's `qualification` suite filtered to this mode (32 shapes,
as of resize/0.11.0's matrix/0.4.0 expansion -- previously only the 2
named performance cases), through kernelforge.benchmark.run_benchmark --
the project's real paired-alternating-blocks protocol. Device
steady-state is the primary metric; graph_replay is reported alongside
it, not substituted for it.

Run: LD_LIBRARY_PATH=/opt/ocv-cuda/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib \
     python3 bench/bench_performance_cubic.py
"""
from __future__ import annotations

import sys

from _common import B, baseline_fn, candidate_fn, cases_for_mode, print_performance_result

MODE = "INTER_CUBIC"


def main() -> int:
    cases = cases_for_mode("qualification", MODE)
    print(f"{MODE} performance -- cases.yaml `qualification` suite ({len(cases)} shapes) "
          f"vs cv::cuda::resize")
    print("=" * 78)
    result = B.run_benchmark(cases, candidate_fn, baseline_fn,
                             baseline_provider="cv::cuda::resize")
    print_performance_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
