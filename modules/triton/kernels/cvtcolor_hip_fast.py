# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright 2026, The FlagOS Contributors.
#
# HIP color fast paths with explicit type and layout predicates.
# The generator appends this fragment to cvtcolor_cpu_hip.py, inheriting its imports.
# Row strides count elements of each kernel's pointer type.


@triton.jit(do_not_specialize=['src', 'dst', 'cols', 'src_rs', 'dst_rs'])
def _cvtcolor_hip_packed(src, dst, cols, src_rs, dst_rs,
                        BLOCK: tl.constexpr):
    """Four aligned uint8 BGR pixels per lane, with independent row pitches."""
    row = tl.program_id(0).to(tl.int64)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = x < cols // 4
    s = (src + row * src_rs).to(tl.pointer_type(tl.uint32))
    d = (dst + row * dst_rs).to(tl.pointer_type(tl.uint32))
    a = tl.load(s + 3 * x, mask, 0)
    b = tl.load(s + 3 * x + 1, mask, 0)
    c = tl.load(s + 3 * x + 2, mask, 0)
    # COLOR_BGR2GRAY, CPU gray_shift=15.
    y0 = ((a & 255) * 3735 + ((a >> 8) & 255) * 19235 + ((a >> 16) & 255) * 9798 + 16384) >> 15
    y1 = ((a >> 24) * 3735 + (b & 255) * 19235 + ((b >> 8) & 255) * 9798 + 16384) >> 15
    y2 = (((b >> 16) & 255) * 3735 + (b >> 24) * 19235 + (c & 255) * 9798 + 16384) >> 15
    y3 = (((c >> 8) & 255) * 3735 + ((c >> 16) & 255) * 19235 + (c >> 24) * 9798 + 16384) >> 15
    tl.store(d + x, y0 | (y1 << 8) | (y2 << 16) | (y3 << 24), mask)


# Four complete BGRA pixels use word loads and one packed grayscale word store.
@triton.jit(do_not_specialize=['src', 'dst', 'cols', 'src_rs', 'dst_rs'])
def _cvtcolor_hip_bgra_gray(src, dst, cols, src_rs, dst_rs,
                           BLOCK: tl.constexpr):
    """Four aligned BGRA byte pixels per lane, preserving CPU integer gray."""
    row = tl.program_id(0).to(tl.int64)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = x < cols // 4
    source = (src + row * src_rs).to(tl.pointer_type(tl.uint32))
    target = (dst + row * dst_rs).to(tl.pointer_type(tl.uint32))
    result = tl.full((BLOCK,), 0, tl.uint32)
    for pixel in tl.static_range(4):
        word = tl.load(source + 4 * x + pixel, mask=mask, other=0)
        gray = ((word & 255) * 3735 + ((word >> 8) & 255) * 19235 +
                ((word >> 16) & 255) * 9798 + 16384) >> 15
        result = result | (gray << (8 * pixel))
    tl.store(target + x, result, mask=mask)


@triton.jit(do_not_specialize=['src', 'dst', 'cols', 'src_rs', 'dst_rs'])
def _cvtcolor_hip_alpha_f32(src, dst, cols, src_rs, dst_rs, BLOCK: tl.constexpr):
    """Copy three float channels and append alpha, with adjacent channel lanes."""
    row = tl.program_id(0).to(tl.int64)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)[:, None]
    channel = tl.arange(0, 4)[None, :]
    active = x < cols
    value = tl.load(src + row * src_rs + x * 3 + channel,
                    mask=active & (channel < 3), other=0.)
    value = tl.where(channel == 3, 1., value)
    tl.store(dst + row * dst_rs + x * 4 + channel, value, mask=active)


@triton.jit(do_not_specialize=['src', 'dst', 'cols', 'src_rs', 'dst_rs'])
def _cvtcolor_hip_gray_f32_c4(src, dst, cols, src_rs, dst_rs,
                             BLOCK: tl.constexpr, BLUE: tl.constexpr):
    """Load adjacent float channels; preserve identified CPU source-order arithmetic."""
    tl.static_assert(BLUE == 0 or BLUE == 2)
    row = tl.program_id(0).to(tl.int64)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    channel = tl.arange(0, 4)
    value = tl.load(src + row * src_rs + x[:, None] * 4 + channel[None, :],
                    mask=x[:, None] < cols, other=0.)
    first = tl.gather(value, tl.full((BLOCK, 1), 0, tl.int32), axis=1).reshape((BLOCK,))
    green = tl.gather(value, tl.full((BLOCK, 1), 1, tl.int32), axis=1).reshape((BLOCK,))
    third = tl.gather(value, tl.full((BLOCK, 1), 2, tl.int32), axis=1).reshape((BLOCK,))
    p0 = _triton_mul_rn(first, 0.114 if BLUE == 0 else 0.299)
    p2 = _triton_mul_rn(third, 0.299 if BLUE == 0 else 0.114)
    gray = _triton_add_rn(_triton_add_rn(p0, _triton_mul_rn(green, 0.587)), p2)
    tl.store(dst + row * dst_rs + x, gray, mask=x < cols)
