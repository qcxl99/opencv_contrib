"""INTER_LINEAR correctness, cases.yaml `core` suite, vs cv::cuda::resize.

Held to `compatibility` (the contract's own per-mode tolerance table,
evaluate_difference), same criterion every other mode's script in this
directory uses -- LINEAR's own compatibility rule is atol<=1
(COMPAT_LINEAR_ATOL), same as AREA/CUBIC, even though LINEAR has no
measured residual in practice (every case run under this criterion is
still expected to land at max_absdiff=0; compatibility is the evaluation
FRAMEWORK applied uniformly across modes here, not a claim that LINEAR
needs the slack). Distribution evidence (mismatch_fraction, mean
absolute error, signed_bias) is still computed and printed on every case
regardless of verdict -- see evaluate_difference's own docstring for why
magnitude alone is not read as sufficient.

Run: LD_LIBRARY_PATH=/opt/ocv-cuda/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib \
     python3 bench/bench_correctness_linear.py
"""
from __future__ import annotations

import sys

from _common import C, K, cases_for_mode, print_correctness_result, resize_cuda_oracle

MODE = "INTER_LINEAR"


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
