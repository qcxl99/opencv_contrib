"""INTER_NEAREST correctness, cases.yaml `core` suite, vs cv::cuda::resize.

Held to strict_exact (atol=0, bit-exact) -- NEAREST is a pure gather, so
even under the project's widened `compatibility` bar it is still required
to match exactly (no rounding path to tolerate a residual on).

Run: LD_LIBRARY_PATH=/opt/ocv-cuda/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib \
     python3 bench/bench_correctness_nearest.py
"""
from __future__ import annotations

import sys

from _common import C, K, cases_for_mode, print_correctness_result, resize_cuda_oracle

MODE = "INTER_NEAREST"


def main() -> int:
    cases = cases_for_mode("core", MODE)
    print(f"{MODE} correctness -- cases.yaml `core` suite, strict_exact")
    print("=" * 78)
    res = C.run_execution_checks(
        K, cases, level="C1", criterion=C.CRITERION_STRICT, oracle=resize_cuda_oracle)
    ok = print_correctness_result(MODE, res)
    print(f"\nTotal {MODE} core cases: {len(cases)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
