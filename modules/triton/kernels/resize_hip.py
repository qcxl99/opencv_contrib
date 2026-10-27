"""Triton resize kernels for the OpenCV triton module -- HIP/hcu backend.

Sibling of resize.py (the CUDA backend kernel), but NOT a semantic port of
it. resize.py targets bit-exact agreement with cv::cuda::resize (no
half-pixel shift, PTX-pinned arithmetic). BW1000 has no CUDA at all, so there
is no cv::cuda::resize on this platform to agree with in the first place --
the only numerical oracle available here is OpenCV CPU resize, and CPU
INTER_LINEAR uses the OPPOSITE coordinate rule (half-pixel-center). Porting
resize.py's no-half-pixel formula verbatim and comparing it against CPU
produced ~99.5% mismatched pixels with maxdiff~170+: that is not rounding
noise, it is two different sampling grids. See the empirical check this file
is based on (comparing cv2.resize output against both candidate formulas on
non-monotonic test data) -- half-pixel matched CPU exactly except for a few
dw=2-style cases off by 1 (round-half-to-even vs round-half-up), which is
within the accepted atol<=1 `compatibility` criterion.

  - No PTX inline-asm equivalent is used (plain Triton arithmetic `a*b`,
    `+`), same reasoning as before: no AMDGCN inline-asm string with the same
    bit-pinning guarantee is available through flagtree's hcu backend, and
    with no cv::cuda::resize to be bit-exact against, atol=0 was never a
    reachable target here regardless.
  - Numerical validation on this platform targets OpenCV CPU resize under the
    `compatibility` criterion (atol<=1, magnitude only, contract resize/0.4.0
    semantics). See knowledge/benchmark_protocol.md and
    operators/resize/reference_notes.md for that criterion's definition.

Algorithm:

  LINEAR -- half-pixel-center rule, matching OpenCV CPU resize
      src_x = (dst_x + 0.5) * fx - 0.5          # float32, WITH half-pixel shift
      x1 = floor(src_x); x2 = x1 + 1
      weights: wx2 = src_x - x1, wx1 = 1 - wx2  (from the UNCLAMPED src_x/x1)
      only the READ index clamps: x1r = clamp(x1, 0, src_w-1), x2r = clamp(x2, 0, src_w-1)
      accumulate in float32, then round-half-to-EVEN and saturate to uint8

  NEAREST -- UNCHANGED from resize.py / cv::cuda::resize's rule, because this
  one already agrees with OpenCV CPU resize (verified empirically: no-half-
  pixel truncation toward zero matches cv2.resize(..., INTER_NEAREST) exactly
  on non-monotonic test data; half-pixel does not). Do not "fix" this to use
  half-pixel coordinates -- that would break it.
      index = trunc(dst_x * fx)                # __float2int_rz
      clamped with min(index, src_w - 1)

Rounding: saturate_cast<uchar>(float) on the reference path is round-half-to-
even. `tl.extra.hip.libdevice.rint` is the hcu backend's equivalent of the
CUDA kernel's `tl.extra.cuda.libdevice.rint` -- same IEEE round-half-to-even,
different namespace.
"""

from __future__ import annotations

import triton
import triton.language as tl


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

    # Source coordinate: half-pixel-center rule, matching OpenCV CPU resize
    # (NOT the CUDA no-half-pixel rule -- see the module docstring for why).
    src_x = (offs_w.to(tl.float32) + 0.5) * fx - 0.5
    src_y = (offs_h.to(tl.float32) + 0.5) * fy - 0.5

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

    # Four weight products, one per corner.
    w11 = wx1[None, :] * wy1[:, None]
    w12 = wx2[None, :] * wy1[:, None]
    w21 = wx1[None, :] * wy2[:, None]
    w22 = wx2[None, :] * wy2[:, None]

    for c in tl.static_range(channels):
        base_11 = y1r[:, None] * src_stride_h + x1r[None, :] * channels + c
        base_12 = y1r[:, None] * src_stride_h + x2r[None, :] * channels + c
        base_21 = y2r[:, None] * src_stride_h + x1r[None, :] * channels + c
        base_22 = y2r[:, None] * src_stride_h + x2r[None, :] * channels + c

        v11 = tl.load(src_ptr + base_11, mask=mask, other=0).to(tl.float32)
        v12 = tl.load(src_ptr + base_12, mask=mask, other=0).to(tl.float32)
        v21 = tl.load(src_ptr + base_21, mask=mask, other=0).to(tl.float32)
        v22 = tl.load(src_ptr + base_22, mask=mask, other=0).to(tl.float32)

        # Plain weighted sum. No instruction-sequence pin on this backend --
        # the hcu compiler is free to reassociate/contract, which is exactly
        # why this targets atol<=1 against OpenCV CPU, not atol=0.
        acc = v11 * w11 + v12 * w12 + v21 * w21 + v22 * w22

        # Round-half-to-even, then saturate.
        rounded = tl.extra.hip.libdevice.rint(acc)
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
    sx = (offs_w.to(tl.float32) * fx).to(tl.int32)
    sy = (offs_h.to(tl.float32) * fy).to(tl.int32)
    sx = tl.minimum(sx, src_w - 1)
    sy = tl.minimum(sy, src_h - 1)

    for c in tl.static_range(channels):
        in_off = sy[:, None] * src_stride_h + sx[None, :] * channels + c
        val = tl.load(src_ptr + in_off, mask=mask, other=0)
        out_off = offs_h[:, None] * dst_stride_h + offs_w[None, :] * channels + c
        tl.store(dst_ptr + out_off, val, mask=mask)


# Starting point only: ported from the CUDA kernel's measured-best config.
# BW1000 has warp_size=64 (vs. NVIDIA's 32) and 80 CUs, so this has NOT been
# re-tuned on this platform -- it is an untuned carry-over, not a measured
# result here. Re-tune before quoting any BW1000 performance number.
DEFAULT_CONFIG = {"BLOCK_H": 1, "BLOCK_W": 256, "num_warps": 4}

VARIANT_ID = "hip_original"
ALGORITHM = ("per-output-pixel gather, 1x256 tile; carried over from the CUDA "
             "kernel's config, not re-tuned for gfx936/warp64")


# NOTE: the Python `resize()` host wrapper is intentionally NOT part of the
# OpenCV module. Only the @triton.jit kernels above are compiled at build time
# and launched from C++, so this file must not import torch.
