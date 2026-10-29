"""INTER_CUBIC correctness, cases.yaml `core` suite, vs cv::cuda::resize.

Held to `compatibility` (the contract's own per-mode tolerance table,
evaluate_difference), NOT strict_exact -- CUBIC has a documented sparse
~0.3-1.3%-of-cases max_absdiff=1 residual that survived full inline-PTX
pinning of every multiply/FMA (contract.yaml), unlike LINEAR/NEAREST/AREA.

Run: LD_LIBRARY_PATH=/opt/ocv-cuda/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib \
     python3 bench/bench_correctness_cubic.py
"""
from __future__ import annotations

import sys

from _common import C, K, cases_for_mode, print_correctness_result, resize_cuda_oracle

MODE = "INTER_CUBIC"


def main() -> int:
    cases = cases_for_mode("core", MODE)
    print(f"{MODE} correctness -- cases.yaml `core` suite, compatibility")
    print("=" * 78)
    res = C.run_execution_checks(
        K, cases, level="C1", criterion=C.CRITERION_COMPAT, oracle=resize_cuda_oracle)
    ok = print_correctness_result(MODE, res)
    print(f"\nTotal {MODE} core cases: {len(cases)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
