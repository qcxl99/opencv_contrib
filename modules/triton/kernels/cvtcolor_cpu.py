# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
#
# CPU-aligned cvtColor kernels for the runtime JIT of modules/triton.
#
# Every conversion here is integer and reproduces the OpenCV CPU result bit for bit; the
# constants and the operation order come from modules/imgproc/src/color*.simd_helpers.hpp and
# color_*.simd.hpp, and each one was checked against a live build over 300k random colours
# with zero mismatches before being written here (runs/cvtcolor-module-0001).
#
# DELIBERATELY ABSENT: HSV2BGR and the HLS pair. Those are float32 in OpenCV, and the same
# build rounds differently in its SIMD body than in its scalar tail (SIMD truncates the
# final *255, the tail rounds). The answer for one pixel therefore depends on where it sits
# in the row, so "bit-exact with the CPU" is not a well-defined target for them. They are
# refused by name rather than given a tolerance that would hide that.
#
# Specialisation policy is the same as the resize kernels: channel counts and the operation
# are constexprs, every other runtime argument is do_not_specialize, so the compile key is
# something C++ can compute exactly.

import triton
import triton.language as tl

# Operation ids. tl.constexpr, not plain ints: Triton rejects a global that is not a
# constexpr inside a kernel, and each id becoming its own binary is what lets the branches
# below fold away instead of costing a runtime compare per pixel.
OP_GRAY = tl.constexpr(0)      # BGR/RGB (3 or 4 ch) -> gray (1 or 4 ch)
OP_PERMUTE = tl.constexpr(1)   # a pure channel gather/insert (BGR<->RGB, add/drop alpha)
OP_YUV = tl.constexpr(2)       # BGR/RGB <-> YUV
OP_YCRCB = tl.constexpr(3)     # BGR/RGB <-> YCrCb
OP_HSV = tl.constexpr(4)       # BGR/RGB -> HSV (integer)

# Every constant a kernel body touches must be a tl.constexpr: Triton rejects a plain
# module global inside @triton.jit. (Annotating with `: tl.constexpr` is NOT the same
# thing and does not work.) Values are OpenCV's own.

# color.simd_helpers.hpp
GRAY_SHIFT = tl.constexpr(15)
BY15 = tl.constexpr(3735)                # == B2YF * 32768 + 0.5
GY15 = tl.constexpr(19235)
RY15 = tl.constexpr(9798)
YUV_SHIFT = tl.constexpr(14)
R2Y = tl.constexpr(4899)                 # == R2YF * 16384
G2Y = tl.constexpr(9617)
B2Y = tl.constexpr(1868)
HSV_SHIFT = tl.constexpr(12)

# color_yuv.simd.hpp: forward chroma and inverse coefficients.
YCRI = tl.constexpr(11682)
YCBI = tl.constexpr(9241)
R2VI = tl.constexpr(14369)
B2UI = tl.constexpr(8061)
V2RI = tl.constexpr(18678)
V2GI = tl.constexpr(-9519)
U2GI = tl.constexpr(-6472)
U2BI = tl.constexpr(33292)
CR2RI = tl.constexpr(22987)
CR2GI = tl.constexpr(-11698)
CB2GI = tl.constexpr(-5636)
CB2BI = tl.constexpr(29049)
CHROMA_DELTA = tl.constexpr(128 << 14)

DEFAULT_CONFIG = {"BLOCK": 256, "num_warps": 4}


@triton.jit
def _cvtcolor_cpu_kernel(src, dst, cols, src_rs, dst_rs, sdiv, hdiv,
                         SCN: tl.constexpr, DCN: tl.constexpr,
                         OP: tl.constexpr, BLUE: tl.constexpr,
                         I0: tl.constexpr, I1: tl.constexpr, I2: tl.constexpr,
                         I3: tl.constexpr, INV: tl.constexpr, HR: tl.constexpr,
                         BLOCK: tl.constexpr):
    """BLOCK output pixels of ONE row per program.

    Two dimensions rather than a flat pass over pixels: a GpuMat's row pitch is not
    cols*channels (OpenCV pads the allocation -- a 4x3 CV_8UC4 GpuMat reports step=512),
    so a flat index must divide by cols to find the row. Addressing row and column
    separately costs nothing and removes the division, and it is the same shape the
    resize kernels use.

    `I0..I3` are the destination channel sources for OP_PERMUTE only (-1 = write 255).
    `BLUE` is the blue index (0 for BGR order, 2 for RGB). `INV` selects the inverse
    direction for OP_YUV / OP_YCRCB. `HR` is the hue range: 180 for BGR2HSV, 256 for
    BGR2HSV_FULL -- the CPU adds hr to a negative hue, so the two differ by this constant
    and not only by the hdiv table.
    """
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = col < cols
    sp = row * src_rs + col * SCN
    dp = row * dst_rs + col * DCN

    if OP == OP_GRAY:
        b = tl.load(src + sp + BLUE, mask=m, other=0).to(tl.int32)
        g = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32)
        r = tl.load(src + sp + (BLUE ^ 2), mask=m, other=0).to(tl.int32)
        y = (b * BY15 + g * GY15 + r * RY15 + (1 << (GRAY_SHIFT - 1))) >> GRAY_SHIFT
        tl.store(dst + dp, y.to(tl.uint8), mask=m)
        if DCN == 4:
            tl.store(dst + dp + 3, tl.full((BLOCK,), 255, tl.int32).to(tl.uint8), mask=m)
    elif OP == OP_PERMUTE:
        if I0 >= 0:
            tl.store(dst + dp + 0, tl.load(src + sp + I0, mask=m, other=0), mask=m)
        else:
            tl.store(dst + dp + 0, tl.full((BLOCK,), 255, tl.int32).to(tl.uint8), mask=m)
        if DCN > 1:
            if I1 >= 0:
                tl.store(dst + dp + 1, tl.load(src + sp + I1, mask=m, other=0), mask=m)
            else:
                tl.store(dst + dp + 1, tl.full((BLOCK,), 255, tl.int32).to(tl.uint8), mask=m)
        if DCN > 2:
            if I2 >= 0:
                tl.store(dst + dp + 2, tl.load(src + sp + I2, mask=m, other=0), mask=m)
            else:
                tl.store(dst + dp + 2, tl.full((BLOCK,), 255, tl.int32).to(tl.uint8), mask=m)
        if DCN > 3:
            if I3 >= 0:
                tl.store(dst + dp + 3, tl.load(src + sp + I3, mask=m, other=0), mask=m)
            else:
                tl.store(dst + dp + 3, tl.full((BLOCK,), 255, tl.int32).to(tl.uint8), mask=m)
    elif OP == OP_YUV or OP == OP_YCRCB:
        if not INV:
            # Forward. Y uses the luma row; the two chroma channels are computed from the
            # ALREADY SHIFTED integer Y, which is what the CPU does and is why this matches.
            b = tl.load(src + sp + BLUE, mask=m, other=0).to(tl.int32)
            g = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32)
            r = tl.load(src + sp + (BLUE ^ 2), mask=m, other=0).to(tl.int32)
            y = (b * B2Y + g * G2Y + r * R2Y + (1 << (YUV_SHIFT - 1))) >> YUV_SHIFT
            if OP == OP_YUV:
                c0 = (r - y) * R2VI
                c1 = (b - y) * B2UI
            else:
                c0 = (r - y) * YCRI
                c1 = (b - y) * YCBI
            c0 = tl.minimum(tl.maximum((c0 + CHROMA_DELTA + (1 << (YUV_SHIFT - 1))) >> YUV_SHIFT, 0), 255)
            c1 = tl.minimum(tl.maximum((c1 + CHROMA_DELTA + (1 << (YUV_SHIFT - 1))) >> YUV_SHIFT, 0), 255)
            tl.store(dst + dp, y.to(tl.uint8), mask=m)
            # c0 is the Cr slot (V for YUV, Cr for YCrCb) and c1 the Cb slot (U, Cb).
            # The CPU writes dst[1 + yuvOrder] = Cr and dst[2 - yuvOrder] = Cb, where
            # yuvOrder is 1 for YUV and 0 for YCrCb -- so YUV is [Y, U, V] and YCrCb is
            # [Y, Cr, Cb]. Writing the Cr slot at +1 unconditionally swapped U and V for
            # YUV; caught by the 179-level mismatch in the first numeric check.
            if OP == OP_YUV:
                tl.store(dst + dp + 1, c1.to(tl.uint8), mask=m)
                tl.store(dst + dp + 2, c0.to(tl.uint8), mask=m)
            else:
                tl.store(dst + dp + 1, c0.to(tl.uint8), mask=m)
                tl.store(dst + dp + 2, c1.to(tl.uint8), mask=m)
        else:
            # Inverse: only the chroma term is rounded and shifted; Y is added afterwards,
            # outside the rounding. Cr/Cb sit in different slots for YUV and YCrCb.
            y = tl.load(src + sp + 0, mask=m, other=0).to(tl.int32)
            if OP == OP_YUV:
                u = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32) - 128
                v = tl.load(src + sp + 2, mask=m, other=0).to(tl.int32) - 128
                bb = u * U2BI
                gg = u * U2GI + v * V2GI
                rr = v * V2RI
            else:
                cr = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32) - 128
                cb = tl.load(src + sp + 2, mask=m, other=0).to(tl.int32) - 128
                bb = cb * CB2BI
                gg = cb * CB2GI + cr * CR2GI
                rr = cr * CR2RI
            h = 1 << (YUV_SHIFT - 1)
            b = y + ((bb + h) >> YUV_SHIFT)
            g = y + ((gg + h) >> YUV_SHIFT)
            r = y + ((rr + h) >> YUV_SHIFT)
            b = tl.minimum(tl.maximum(b, 0), 255)
            g = tl.minimum(tl.maximum(g, 0), 255)
            r = tl.minimum(tl.maximum(r, 0), 255)
            tl.store(dst + dp + BLUE, b.to(tl.uint8), mask=m)
            tl.store(dst + dp + 1, g.to(tl.uint8), mask=m)
            tl.store(dst + dp + (BLUE ^ 2), r.to(tl.uint8), mask=m)
            if DCN == 4:
                tl.store(dst + dp + 3, tl.full((BLOCK,), 255, tl.int32).to(tl.uint8), mask=m)
    else:
        # OP_HSV, forward only: the integer path of RGB2HSV_b.
        b = tl.load(src + sp + BLUE, mask=m, other=0).to(tl.int32)
        g = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32)
        r = tl.load(src + sp + (BLUE ^ 2), mask=m, other=0).to(tl.int32)
        v = tl.maximum(tl.maximum(b, g), r)
        mn = tl.minimum(tl.minimum(b, g), r)
        d = v - mn
        s = (d * tl.load(sdiv + v, mask=m, other=0) + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT
        # Channel priority r > g > b, as the CPU's vr/vg masks encode.
        hraw = tl.where(v == r, g - b, tl.where(v == g, b - r + 2 * d, r - g + 4 * d))
        h = (hraw * tl.load(hdiv + d, mask=m, other=0) + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT
        h = tl.where(h < 0, h + HR, h)
        h = tl.minimum(tl.maximum(h, 0), 255)
        tl.store(dst + dp, h.to(tl.uint8), mask=m)
        tl.store(dst + dp + 1, s.to(tl.uint8), mask=m)
        tl.store(dst + dp + 2, v.to(tl.uint8), mask=m)
