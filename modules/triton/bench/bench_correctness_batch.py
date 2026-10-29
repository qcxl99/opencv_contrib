"""resize_batch() correctness: each batch item vs the per-item oracle.

No cases.yaml entry exists for batch (different calling convention from
every per-image Case this project's cases.yaml describes -- a batch of
N images has no single-image Case equivalent), so the shape list here is
this script's own, matching the ones bench_full_resize.py used before
the split (one LINEAR C3, one NEAREST C1, one LINEAR C1, spanning both
dtype-eligible channel counts and two of the four interpolation modes
resize_batch() supports).

Run: LD_LIBRARY_PATH=/opt/ocv-cuda/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib \
     python3 bench/bench_correctness_batch.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from _common import K, resize_cuda_oracle

BATCH_CASES = [
    ("batch_linear_n8_c3", "INTER_LINEAR", 8, 480, 640, 240, 320, 3),
    ("batch_nearest_n32_c1", "INTER_NEAREST", 32, 128, 128, 64, 64, 1),
    ("batch_linear_n16_c1", "INTER_LINEAR", 16, 720, 1280, 360, 640, 1),
]


def main() -> int:
    print("resize_batch() correctness -- per-item vs cv::cuda::resize oracle")
    print("=" * 78)
    all_ok = True
    for name, mode, n, sh, sw, dh, dw, ch in BATCH_CASES:
        shape = (n, sh, sw) if ch == 1 else (n, sh, sw, ch)
        rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
        src = torch.from_numpy(rng.integers(0, 256, shape, dtype=np.uint8)).cuda()
        out = K.resize_batch(src, (dw, dh), mode)
        torch.cuda.synchronize()

        max_diff = 0
        for i in range(n):
            ref = resize_cuda_oracle(src[i], dw, dh, mode)
            torch.cuda.synchronize()
            max_diff = max(max_diff, int((out[i].int() - ref.int()).abs().max()))
        ok = max_diff == 0
        all_ok &= ok
        print(f"  {name:24s} n={n:3d} max_absdiff={max_diff} {'OK' if ok else 'FAIL'}")

    print(f"\n{'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
