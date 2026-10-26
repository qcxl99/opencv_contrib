"""Triton resize kernels for the OpenCV triton module.

Goal for this version is clarity and correctness, not speed. It is allowed to
be slower than cv::cuda::resize; it exists to be the fixed research baseline
that later candidates are compared against.

Semantics mirror the CUDA kernel exactly (cudawarping/src/cuda/resize.cu):

  LINEAR  (resize.cu:234)
      src_x = dst_x * fx                       # float32, NO half-pixel shift
      x1 = floor(src_x); x2 = x1 + 1
      x2_read = min(x2, src_w - 1)             # only the READ index clamps
      weights use the UNCLAMPED x2:  (x2 - src_x) and (src_x - x1)
      accumulate in float32, then round-half-to-EVEN and saturate to uint8

  NEAREST (resize.cu:220)
      index = trunc(dst_x * fx)                # __float2int_rz
      clamped with min(index, src_w - 1)

Why the weights use the unclamped x2: at the right/bottom edge the kernel
reads the clamped pixel but weights it as though the out-of-range neighbour
existed. Dropping that detail changes edge pixels, so it is reproduced rather
than "cleaned up".

Rounding: saturate_cast<uchar>(float) lowers to PTX cvt.rni.sat.u8.f32, i.e.
round-to-nearest-EVEN. Triton has no direct rni intrinsic for this, so it is
built from libdevice rint, which is the same IEEE round-half-to-even.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _fmul(a, b):
    """Single mul.f32, not reassociable by the compiler.

    Needed because plain `a * b` in Triton gets folded into surrounding FMAs,
    which changes the rounding. See the note on _resize_linear_kernel.
    """
    return tl.inline_asm_elementwise(
        "mul.f32 $0, $1, $2;", "=f,f,f", [a, b],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _ffma(a, b, c):
    """Single fma.rn.f32, matching the CUDA kernel's accumulation chain."""
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", [a, b, c],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _resize_linear_kernel(
    src_ptr, dst_ptr,
    src_h, src_w, dst_h, dst_w,
    src_stride_h, dst_stride_h,
    channels: tl.constexpr,
    fx, fy,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """One program handles a BLOCK_H x BLOCK_W tile of the output."""
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_h = offs_h < dst_h
    mask_w = offs_w < dst_w
    mask = mask_h[:, None] & mask_w[None, :]

    # Source coordinate: dst * f, with no half-pixel shift (CUDA rule).
    # _fmul pins a single mul.f32 so the compiler cannot fold it elsewhere.
    src_x = _fmul(offs_w.to(tl.float32), fx)
    src_y = _fmul(offs_h.to(tl.float32), fy)

    x1 = tl.floor(src_x).to(tl.int32)
    y1 = tl.floor(src_y).to(tl.int32)
    x2 = x1 + 1
    y2 = y1 + 1

    # Weights come from the UNCLAMPED neighbour index.
    wx2 = src_x - x1.to(tl.float32)
    wx1 = x2.to(tl.float32) - src_x
    wy2 = src_y - y1.to(tl.float32)
    wy1 = y2.to(tl.float32) - src_y

    # Only the read index is clamped.
    x1r = tl.minimum(tl.maximum(x1, 0), src_w - 1)
    x2r = tl.minimum(x2, src_w - 1)
    y1r = tl.minimum(tl.maximum(y1, 0), src_h - 1)
    y2r = tl.minimum(y2, src_h - 1)

    # Precompute the four weight products, exactly as the CUDA kernel does.
    # Verified against the compiled SASS of resize_linear<uchar>: four FMULs
    # produce the weights, then a chain of four FFMAs accumulates
    # out = fma(v, w, out) starting from zero.
    #
    # Plain Triton arithmetic will NOT reproduce this. Writing `acc += v * w`
    # (or any algebraically equivalent separable lerp) lets the compiler
    # reassociate and contract the operations: the generated PTX had 14
    # fma.rn.f32 and zero add.f32, and the accumulator came out as
    # 105.499985 where exact float32 gives 105.5 -- a tie that rounds to 106.
    # That cost 1 LSB on ~0.1% of pixels across 65 of 150 random cases, and
    # three different formulations all failed identically. Pinning the exact
    # instruction sequence with inline PTX brought it to 0 of 100.
    w11 = _fmul(wx1[None, :], wy1[:, None])
    w12 = _fmul(wx2[None, :], wy1[:, None])
    w21 = _fmul(wx1[None, :], wy2[:, None])
    w22 = _fmul(wx2[None, :], wy2[:, None])

    for c in tl.static_range(channels):
        base_11 = y1r[:, None] * src_stride_h + x1r[None, :] * channels + c
        base_12 = y1r[:, None] * src_stride_h + x2r[None, :] * channels + c
        base_21 = y2r[:, None] * src_stride_h + x1r[None, :] * channels + c
        base_22 = y2r[:, None] * src_stride_h + x2r[None, :] * channels + c

        v11 = tl.load(src_ptr + base_11, mask=mask, other=0).to(tl.float32)
        v12 = tl.load(src_ptr + base_12, mask=mask, other=0).to(tl.float32)
        v21 = tl.load(src_ptr + base_21, mask=mask, other=0).to(tl.float32)
        v22 = tl.load(src_ptr + base_22, mask=mask, other=0).to(tl.float32)

        # Four-term FFMA chain, same order and same fusion as the CUDA kernel.
        acc = _ffma(v11, w11, tl.zeros_like(v11))
        acc = _ffma(v12, w12, acc)
        acc = _ffma(v21, w21, acc)
        acc = _ffma(v22, w22, acc)

        # cvt.rni.sat.u8.f32: round-half-to-even, then saturate.
        rounded = tl.extra.cuda.libdevice.rint(acc)
        clamped = tl.minimum(tl.maximum(rounded, 0.0), 255.0)

        out_off = offs_h[:, None] * dst_stride_h + offs_w[None, :] * channels + c
        tl.store(dst_ptr + out_off, clamped.to(tl.uint8), mask=mask)


@triton.jit
def _resize_nearest_kernel(
    src_ptr, dst_ptr,
    src_h, src_w, dst_h, dst_w,
    src_stride_h, dst_stride_h,
    channels: tl.constexpr,
    fx, fy,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (offs_h < dst_h)[:, None] & (offs_w < dst_w)[None, :]

    # __float2int_rz == truncate toward zero. Coordinates are non-negative
    # here, so floor and trunc agree, but trunc is what the kernel specifies.
    sx = _fmul(offs_w.to(tl.float32), fx).to(tl.int32)
    sy = _fmul(offs_h.to(tl.float32), fy).to(tl.int32)
    sx = tl.minimum(sx, src_w - 1)
    sy = tl.minimum(sy, src_h - 1)

    for c in tl.static_range(channels):
        in_off = sy[:, None] * src_stride_h + sx[None, :] * channels + c
        val = tl.load(src_ptr + in_off, mask=mask, other=0)
        out_off = offs_h[:, None] * dst_stride_h + offs_w[None, :] * channels + c
        tl.store(dst_ptr + out_off, val, mask=mask)


# Launch configuration selected by measurement, not by intuition. Evidence:
# every variant holding 32 registers sits at exactly 6 elements/thread; pushing
# to 24 or 48 elements/thread drove registers to 108/201 and collapsed achieved
# occupancy to 22%, costing 2-3x in latency. See
# knowledge/diagnostics_c3_bottleneck.md. The original 8x32 config is preserved
# as runs/opt-0001/candidates/v001_original.py, which remains the fixed
# research reference.
DEFAULT_CONFIG = {"BLOCK_H": 1, "BLOCK_W": 256, "num_warps": 4}

VARIANT_ID = "original"
ALGORITHM = "per-output-pixel gather, 1x256 tile (autotuned: best single config across perf cases)"


# NOTE: the Python `resize()` host wrapper is intentionally NOT part of the
# OpenCV module. Only the @triton.jit kernels above are compiled at build time
# and launched from C++, so this file must not import torch.
