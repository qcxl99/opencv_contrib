"""INTER_AREA correctness, cases.yaml `core` suite, vs cv::cuda::resize.

Held to `compatibility` by the contract's own per-mode tolerance table
(evaluate_difference), the same bar CUBIC uses -- NOT strict_exact, though
AREA is measured bit-exact on every case tried so far (contract.yaml's own
note: "AREA IS [exact], measured bit-exact on every case tried"). Includes
cases exercising both filter branches (IntegerAreaFilter / AreaFilter) and
the upscale-or-identity-on-either-axis aliasing-to-LINEAR dispatch quirk.

Run: LD_LIBRARY_PATH=/opt/ocv-cuda/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib \
     python3 bench/bench_correctness_area.py
"""
from __future__ import annotations

import sys

from _common import C, K, cases_for_mode, print_correctness_result, resize_cuda_oracle

MODE = "INTER_AREA"


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
