"""resize_batch() performance vs a loop of single-image resize() calls.

No native baseline exists for this comparison -- cv::cuda::resize has no
batch concept at all, so there is nothing to measure "vs native" here.
What IS measurable is whether the batch calling convention itself saves
launch/dispatch overhead over calling resize() N times, which is the
comparison this script makes: resize_batch() vs an equivalent loop of
single-image resize() calls, same total work, same data.

Timed via paired_alternating_graph_replay (_common.py): alternates which
side runs first across repeats and takes the median, the same two
disciplines kernelforge.benchmark.benchmark_case() applies to every
other comparison in this project -- not a single sequential measurement
(bench_p2_resize.py's run_batch_case() used a non-alternating
_device_block_ms call per side; this retires that).

Run: LD_LIBRARY_PATH=/opt/ocv-cuda/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib \
     python3 bench/bench_performance_batch.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from _common import K, paired_alternating_graph_replay

BATCH_CASES = [
    ("batch_linear_n8_c3", "INTER_LINEAR", 8, 480, 640, 240, 320, 3),
    ("batch_nearest_n32_c1", "INTER_NEAREST", 32, 128, 128, 64, 64, 1),
    ("batch_linear_n16_c1", "INTER_LINEAR", 16, 720, 1280, 360, 640, 1),
]

BLOCK_ITERS = 50
GRAPH_BATCH = 20


def main() -> int:
    print("resize_batch() performance -- vs a loop of single-image resize() calls")
    print("=" * 78)
    for name, mode, n, sh, sw, dh, dw, ch in BATCH_CASES:
        shape = (n, sh, sw) if ch == 1 else (n, sh, sw, ch)
        dshape = (n, dh, dw) if ch == 1 else (n, dh, dw, ch)
        rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
        src = torch.from_numpy(rng.integers(0, 256, shape, dtype=np.uint8)).cuda()
        dst = torch.empty(dshape, dtype=torch.uint8, device="cuda")
        single_dst = torch.empty((dh, dw) if ch == 1 else (dh, dw, ch),
                                 dtype=torch.uint8, device="cuda")
        st = torch.cuda.Stream()

        def batched():
            K.resize_batch(src, (dw, dh), mode, stream=st, out=dst)

        def loop_of_singles():
            for i in range(n):
                K.resize(src[i], (dw, dh), mode, stream=st, out=single_dst)

        with torch.cuda.stream(st):
            for _ in range(20):
                batched()
                loop_of_singles()
        st.synchronize()

        lo, hi, med = paired_alternating_graph_replay(
            batched, loop_of_singles, st, BLOCK_ITERS, GRAPH_BATCH)
        if med is None:
            print(f"  {name:24s} ALL_GRAPH_CAPTURE_FAILED")
            continue
        print(f"  {name:24s} n={n:3d} speedup(loop/batch)=[{lo:.3f},{hi:.3f}] "
              f"median={med:.3f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
