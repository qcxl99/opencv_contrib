#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

# Copyright 2026, The FlagOS Contributors.
"""Generate HIP kernel sources from the module's CUDA-compatible sources.

Run without arguments to write kernels/<op>_cpu_hip.py, or use --check to verify
that the checked-in files match the current sources and this generator.

The source kernels use libdevice operations to preserve floating-point rounding.
The HIP libdevice implementation does not provide every operation, so this tool
substitutes the shims below. Kernels without those imports can be used directly.
Resize appends kernels/resize_hip_fast.py; cvtColor appends kernels/cvtcolor_hip_fast.py.
Edit those maintained sources, not their copies in generated files.

Each generated file records the digest of its local source for traceability.
"""
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.join(HERE, "..", "kernels")
OPS = ("resize", "cvtcolor", "warpaffine")

ANCHOR = "from triton.language.extra import libdevice\n"

# Common resize arithmetic shim. Keeping this block literal makes the generated
# source reproducible without importing Triton or a backend compiler.
_RESIZE_SHIM = """\


# HIP/AMDGCN replacements for missing libdevice primitives.
# Explicit vector add/multiply instructions prevent multiply-add contraction.
# Division uses Triton's `/` lowering; it is not a correctly rounded div_rn primitive.
@triton.jit
def _triton_add_rn(a, b):
    return tl.inline_asm_elementwise("v_add_f32 $0, $1, $2", "=v,v,v", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _triton_mul_rn(a, b):
    return tl.inline_asm_elementwise("v_mul_f32 $0, $1, $2", "=v,v,v", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _triton_div_rn(a, b):
    return a / b

"""

# Per-primitive bodies, for a kernel whose missing set differs from resize's.
_SHIMS = {
    "float2int_rn": """
@triton.jit
def _triton_float2int_rn(a):
    # The packed byte interpolation range is finite and inside [0, 255].
    rounded = tl.inline_asm_elementwise("v_rndne_f32 $0, $1", "=v,v", [a],
                                        dtype=tl.float32, is_pure=True, pack=1)
    return rounded.to(tl.int32)

""",

    "rint": """
@triton.jit
def _triton_rint(a):
    return tl.inline_asm_elementwise("v_rndne_f32 $0, $1", "=v,v", [a], dtype=tl.float32, is_pure=True, pack=1)

""",
    "add_rn": """
@triton.jit
def _triton_add_rn(a, b):
    return tl.inline_asm_elementwise("v_add_f32 $0, $1, $2", "=v,v,v", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)

""",
    "mul_rn": """
@triton.jit
def _triton_mul_rn(a, b):
    return tl.inline_asm_elementwise("v_mul_f32 $0, $1, $2", "=v,v,v", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)

""",
    "sub_rn": """
@triton.jit
def _triton_sub_rn(a, b):
    return tl.inline_asm_elementwise("v_sub_f32 $0, $1, $2", "=v,v,v", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)

""",
    "div_rn": """
@triton.jit
def _triton_div_rn(a, b):
    return a / b
""",
}

# Fixed order: two kernels missing the same primitives derive identical text.
MISSING_ON_HIP = ("add_rn", "mul_rn", "div_rn", "sub_rn", "rint", "float2int_rn")
RESIZE_SET = ("add_rn", "mul_rn", "div_rn")


def shim_for(text: str) -> str:
    """The shim block for the primitives `text` calls and flagtree's hip lacks.

    The common resize set uses a fixed block. Other sets are composed in the
    order of MISSING_ON_HIP and name the primitives they provide.
    """
    used = tuple(n for n in MISSING_ON_HIP if re.search(rf"\blibdevice\.{n}\(", text))
    if not used:
        return ""
    if used == RESIZE_SET:
        return _RESIZE_SHIM
    arith = [n for n in used if n not in ("div_rn", "rint", "float2int_rn")]
    head = ("\n\n# HIP/AMDGCN replacements for missing libdevice primitives: "
            + ", ".join(used) + ".\n")
    if arith:
        head += "# Explicit vector arithmetic preserves separate add/multiply/subtract operations.\n"
    if "rint" in used:
        head += "# v_rndne_f32 rounds to the nearest even integer.\n"
    if "div_rn" in used:
        head += "# Division uses Triton's `/` lowering, not a correctly rounded div_rn primitive.\n"
    return head + "".join(_SHIMS[n] for n in used)


def paths(op):
    return (os.path.join(KERNELS, f"{op}_cpu.py"), os.path.join(KERNELS, f"{op}_cpu_hip.py"))


def needs_derivation(op):
    src = paths(op)[0]
    return os.path.isfile(src) and ANCHOR in open(src, encoding="utf-8").read()


def reuse_cubic_rows(body):
    """Retain overlapping horizontal rows when downsampling advances by two or three."""
    old = """                else:
                    h0 = _cpu_hcub(_cpu_row(base, sy, -1, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)"""
    new = """                elif sy == prev + 2:
                    # The new four-row window overlaps the last two cached rows.
                    h0 = h2
                    h1 = h3
                    h2 = _cpu_hcub(_cpu_row(base, sy, 1, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)
                    h3 = _cpu_hcub(_cpu_row(base, sy, 2, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)
                elif sy == prev + 3:
                    h0 = h3
                    h1 = _cpu_hcub(_cpu_row(base, sy, 0, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)
                    h2 = _cpu_hcub(_cpu_row(base, sy, 1, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)
                    h3 = _cpu_hcub(_cpu_row(base, sy, 2, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)
                else:
                    h0 = _cpu_hcub(_cpu_row(base, sy, -1, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)"""
    if body.count(old) != 1:
        raise ValueError("resize cubic row-cache anchor changed")
    return body.replace(old, new, 1)


def aligned_cubic_tail(body):
    """Skip the unused fixed-point tail on complete eight-element byte rows."""
    # Uniform row alignment makes the CPU scalar-tail result unused.
    start = body.index("                # ... scalar tail: FixedPtCast<int, uchar, 22>")
    end = body.index("            else:", start)
    tail = body[start:end]
    body = body[:start] + ("                if W % 8 == 0:\n"
                           "                    out = vf.to(tl.uint8)\n"
                           "                else:\n" +
                           "\n".join("    " + line for line in tail.rstrip().split("\n")) + "\n") + body[end:]
    return body


def cubic_row_reuse_kernel(body):
    """Derive a separate row-cache variant; upscales retain the original control flow."""
    function_start = body.index("def _resize_cpu_sep_kernel(")
    start = body.rfind("@triton.jit", 0, function_start)
    end = body.index("\n\n@triton.jit", function_start)
    function = reuse_cubic_rows(body[start:end])
    return function.replace("def _resize_cpu_sep_kernel(", "def _resize_cpu_sep_reuse_kernel(", 1)




def warp_i32_channel_kernel(body):
    """Keep cubic arithmetic; the host bounds both byte spans before using i32 offsets."""
    start = body.index("@triton.jit\ndef _warp_cubic_hip_channels(")
    end = body.find("\n\n@triton.jit", start + 1)
    if end < 0:
        end = len(body)
    part = body[start:end].replace("def _warp_cubic_hip_channels(",
                                   "def _warp_cubic_i32_hip_channels(", 1)
    old = "idx = (cy.to(tl.int64) * src_rs + cx.to(tl.int64) * cn + c)"
    if part.count(old) != 1:
        raise ValueError("cubic channel source offset anchor changed")
    part = part.replace(old, "idx = (cy * src_rs + cx * cn + c).to(tl.int64)")
    old = "dst + ys.to(tl.int64) * dst_rs + xs.to(tl.int64) * cn + c"
    if old not in part:
        raise ValueError("cubic channel destination offset anchor changed")
    return part.replace(old, "dst + (ys * dst_rs + xs * cn + c).to(tl.int64)")


def packed_u8_cubic_kernel(body):
    """Share aligned RGBA words while retaining every canonical cubic operation."""
    start = body.index("@triton.jit\ndef _warp_cubic(")
    end = body.index("\n\n@triton.jit", start + 1)
    host = body.find("\n\ndef ", start + 1)
    if host >= 0:
        end = min(end, host)
    part = body[start:end].replace("def _warp_cubic(", "def _warp_cubic_packed4_u8(", 1)
    loop = "    for c in tl.static_range(cn):"
    if part.count(loop) != 1:
        raise ValueError("cubic channel loop anchor changed")
    part = part.replace(loop,
        "    tl.static_assert(U8 and not U16 and cn == 4 and BORDER == 1 and not TRANSPARENT)\n"
        "    packed = tl.zeros((BLOCK_H, BLOCK_W), tl.uint32)\n" + loop, 1)
    old = "idx = (cy.to(tl.int64) * src_rs + cx.to(tl.int64) * cn + c)"
    if part.count(old) != 1:
        raise ValueError("cubic source word offset anchor changed")
    part = part.replace(old, "idx = cy.to(tl.int64) * (src_rs // 4) + cx.to(tl.int64)")
    old = "                    v = tl.load(src + idx, mask=m, other=0)"
    if part.count(old) != 1:
        raise ValueError("cubic source word load anchor changed")
    part = part.replace(old,
        "                    word = tl.load(src.to(tl.pointer_type(tl.uint32)) + idx, mask=m, other=0)\n"
        "                    v = (word >> (8 * c)) & 255")
    old = "        tl.store(dst + ys.to(tl.int64) * dst_rs + xs.to(tl.int64) * cn + c, out, mask=cstore)"
    if part.count(old) != 1:
        raise ValueError("cubic destination word store anchor changed")
    part = part.replace(old,
        "        packed = packed | (out.to(tl.uint32) << (8 * c))\n"
        "    tl.store(dst.to(tl.pointer_type(tl.uint32)) + ys.to(tl.int64) * (dst_rs // 4) + xs.to(tl.int64),\n"
        "             packed, mask=m)")
    return part.replace('"""INTER_CUBIC:',
                        '"""Aligned U8 C4 word access; canonical cubic arithmetic.\n\n    INTER_CUBIC:', 1)


def halfpixel_cubic_kernel(body, name):
    """Specialize only exact half-pixel translations admitted by the host."""
    start = body.index("@triton.jit\ndef " + name + "(")
    end = body.find("\n\n@triton.jit", start + 1)
    if end < 0:
        end = len(body)
    host = body.find("\n\ndef ", start + 1)
    if host >= 0:
        end = min(end, host)
    part = body[start:end].replace("def " + name + "(", "def " + name + "_halfpixel(", 1)
    for axis in ("x", "y"):
        old = "    w{0}0, w{0}1, w{0}2, w{0}3 = _bicubic_weights_t(a{0}, FUSED)".format(axis)
        if part.count(old) != 1:
            raise ValueError("cubic weight anchor changed: " + name + "/" + axis)
        part = part.replace(old,
            "    w{0}0, w{0}1, w{0}2, w{0}3 = -0.09375, 0.59375, 0.59375, -0.09375".format(axis))
    return part.replace('"""',
        '"""Host admits unit linear part, matching sizes below 2**23 and exact +/-0.5 shifts.\n    ', 1)

def warp_channel_kernel(body, mode="linear"):
    """Derive a C3/C4 pixel/channel tile without changing coordinate or tap arithmetic."""
    start = body.index("@triton.jit\ndef _warp_" + mode + "(")
    end = body.index("\n\n@triton.jit", start + 1)
    # Host helpers can follow the last JIT kernel before an appended HIP kernel.
    next_host = body.find("\n\ndef ", start + 1)
    if next_host >= 0:
        end = min(end, next_host)
    function = body[start:end].replace("def _warp_" + mode + "(", "def _warp_" + mode + "_hip_channels(", 1)
    function = function.replace("[:, None]", "[:, None, None]", 1)
    function = function.replace("[None, :]", "[None, :, None]", 1)
    head, channel_body = function.split("    for c in tl.static_range(cn):\n", 1)
    channel_body = "\n".join(line[4:] if line.startswith("    ") else line
                              for line in channel_body.split("\n"))
    old = """        if c == 0:
            bv = bv0
        elif c == 1:
            bv = bv1
        elif c == 2:
            bv = bv2
        else:
            bv = bv3"""
    channel_body = channel_body.replace(old,
        "        bv = tl.where(c == 0, bv0, tl.where(c == 1, bv1, tl.where(c == 2, bv2, bv3)))")
    channel_body = channel_body.replace("tl.full(v.shape, bv.to(v.dtype), v.dtype)", "bv.to(v.dtype)")
    channel_body = channel_body.replace("(BLOCK_H, BLOCK_W)", "(BLOCK_H, BLOCK_W, 4)")
    # Nested cubic tap fetches also need a vector border value. Keep the
    # canonical scalar-channel kernels unchanged.
    border_pattern = (r"(?m)^([ ]+)if c == 0:\n\1    bv = bv0\n"
                      r"\1elif c == 1:\n\1    bv = bv1\n"
                      r"\1elif c == 2:\n\1    bv = bv2\n"
                      r"\1else:\n\1    bv = bv3")
    channel_body = re.sub(border_pattern,
                          lambda match: match[1] + "bv = tl.where(c == 0, bv0, tl.where(c == 1, bv1, tl.where(c == 2, bv2, bv3)))",
                          channel_body)
    channel_body = channel_body.replace("tl.full(acc.shape, bv.to(tl.float32), tl.float32)",
                                        "bv.to(tl.float32)")
    setup = ("    tl.static_assert((cn == 3 or cn == 4) and BORDER >= 0 and BORDER <= 5)\n"
             "    c = tl.arange(0, 4)[None, None, :]\n"
             "    m = m & (c < cn)\n")
    return head + setup + channel_body


def compact_luv_lookup(body):
    """Read the CPU table's duplicated corners in six aligned machine words."""
    start = body.index("            # Steps to the corner one node up")
    end = body.index("            # Channel stride inside a node", start)
    body = body[:start] + body[end:]
    start = body.index("            zero = tl.zeros((BLOCK,), tl.int32)", start)
    end = body.index("                pr = (i >> 2) & 1", start)
    body = body[:start] + "            for i in tl.static_range(8):\n" + body[end:]
    body = body.replace("                off = px + (s_r if pr else zero) + (s_g if qg else zero) + (s_b if rb else zero)\n", "", 1)
    anchor = "            accL = tl.zeros((BLOCK,), tl.int32)"
    preload = """            # labluvTablesFor packs the internal table at 256-byte alignment.
            # One node contains 24 int16 values: two words per output channel.
            words = t16.to(tl.pointer_type(tl.uint64))
            word_base = px // 4
            l0 = tl.load(words + word_base, mask=m, other=0)
            l1 = tl.load(words + word_base + 1, mask=m, other=0)
            u0 = tl.load(words + word_base + 2, mask=m, other=0)
            u1 = tl.load(words + word_base + 3, mask=m, other=0)
            v0 = tl.load(words + word_base + 4, mask=m, other=0)
            v1 = tl.load(words + word_base + 5, mask=m, other=0)
"""
    if body.count(anchor) != 1:
        raise ValueError("Luv accumulator anchor changed")
    body = body.replace(anchor, preload + anchor, 1)
    old = """                accL += tl.load(t16 + off, mask=m, other=0) * wgt
                accU += tl.load(t16 + off + 8, mask=m, other=0) * wgt
                accV += tl.load(t16 + off + 16, mask=m, other=0) * wgt"""
    new = """                shift = (i % 4) * 16
                lv = ((l0 if i < 4 else l1) >> shift).to(tl.int16).to(tl.int32)
                uv = ((u0 if i < 4 else u1) >> shift).to(tl.int16).to(tl.int32)
                vv = ((v0 if i < 4 else v1) >> shift).to(tl.int16).to(tl.int32)
                accL += lv * wgt
                accU += uv * wgt
                accV += vv * wgt"""
    if body.count(old) != 1:
        raise ValueError("Luv channel load anchor changed")
    return body.replace(old, new, 1)


def reconstruct_luv_inverse(body):
    """Reconstruct the exact LvToVpl table value from the already loaded LvToV value."""
    old = "            vpl = tl.load(t64 + LL * 256 + vv, mask=m, other=0)"
    new = """            # CPU initLUTforLUV defines LvToVpl from LvToV * L. The
            # first product fits int32; widen before the larger constant multiply.
            vpl = (vp * LL).to(tl.int64) * (12 * 13 * 100 * (BASE // 1024))"""
    if body.count(old) != 1:
        raise ValueError("Luv inverse table anchor changed")
    body = body.replace(old, new, 1)
    # Clamping negative z to zero can precede positive constant divisions.
    # Keep the original signed-int32 cast before max: unsigned-first is wrong.
    old_x = """            x = (xv // BASE).to(tl.int32)
            x = ((y64 * x.to(tl.int64)) // BASE).to(tl.int32)"""
    new_x = """            # The first quotient fits int32 for every LuToUp/LvToVp pair.
            # Since Y is nonnegative, a negative X is clipped to zero at the end.
            positive_x = tl.maximum(xv, 0).to(tl.uint64)
            x = (positive_x >> 14).to(tl.uint32)
            x = ((y.to(tl.uint64) * x.to(tl.uint64)) >> 14).to(tl.int32)"""
    if body.count(old_x) != 1:
        raise ValueError("Luv inverse x anchor changed")
    body = body.replace(old_x, new_x, 1)
    old_z = "            z = zm // 256 + zm // 65536"
    new_z = """            positive_z = tl.maximum(zm, 0).to(tl.uint32)
            z = ((positive_z >> 8) + (positive_z >> 16)).to(tl.int32)"""
    if body.count(old_z) != 1:
        raise ValueError("Luv inverse z anchor changed")
    return body.replace(old_z, new_z, 1)


def aligned_f32_cubic_kernel(body):
    """Eliminate the scalar vertical tail only for a host-admitted complete float row."""
    name = "_resize_cpu_sep_kernel"
    function_start = body.index("def " + name + "(")
    start = body.rfind("@triton.jit", 0, function_start)
    end = body.find("\n\n@triton.jit", function_start)
    if end < 0:
        end = len(body)
    function = body[start:end]
    new_name = "_resize_cpu_sep_f32_aligned_kernel"
    function = function.replace("def " + name + "(", "def " + new_name + "(", 1)
    old = """                tail = _triton_add_rn(_triton_add_rn(_triton_add_rn(p0, p1), p2), p3)
                out = tl.where(e < (W // 4) * 4, vec, tail)"""
    if function.count(old) != 1:
        raise ValueError("float cubic vertical tail anchor changed")
    function = function.replace(old, "                out = vec", 1)
    anchor = "    # xofs/yofs are source base indices; CUBIC starts one pixel/row before the base."
    if function.count(anchor) != 1:
        raise ValueError("separable kernel precondition anchor changed")
    return function.replace(anchor, "    tl.static_assert(not U8 and KSIZE == 4)\n" + anchor, 1)


def packed_u16_nearest(body):
    """Copy one aligned ushort C4 pixel per lane with the original coordinates."""
    start = body.index("@triton.jit\ndef _warp_nearest_packed4(")
    end = body.index("\n\n@triton.jit", start + 1)
    part = body[start:end].replace("_warp_nearest_packed4(", "_warp_nearest_packed4_u16(", 1)
    border_start = part.index("    border = (")
    border_end = part.index("    _warp_nearest(", border_start)
    part = part[:border_start] + "    tl.static_assert(BORDER == 1 and not TRANSPARENT)\n    border = 0\n" + part[border_end:]
    return part.replace("tl.uint32", "tl.uint64").replace("aligned uint8 C4", "aligned uint16 C4")


def static_area_kernel(body):
    """Specialize only bounded table tap counts; keep every ordered sum and mask."""
    start = body.index("@triton.jit(do_not_specialize=", body.index("def _resize_cpu_area_fast_kernel"))
    end = body.find("\n\n@triton.jit", start + 1)
    if end < 0:
        end = len(body)
    part = body[start:end]
    if "def _resize_cpu_area_kernel(" not in part:
        raise ValueError("generic area derivation anchor changed")
    part = part.replace("_resize_cpu_area_kernel(", "_resize_cpu_area_static_kernel(", 1)
    part = part.replace("'KX', 'KY', ", "", 1)
    part = part.replace("ys, ya, yc, KX, KY,", "ys, ya, yc, KX: tl.constexpr, KY: tl.constexpr,", 1)
    part = part.replace("for ky in range(0, KY):", "for ky in tl.static_range(0, KY):", 1)
    part = part.replace("for kx in tl.range(0, KX, loop_unroll_factor=2):", "for kx in tl.static_range(0, KX):", 1)
    return part


def packed_f32_nearest(body):
    """Copy a C4 float pixel as two aligned 64-bit words; preserve coordinate math."""
    start = body.index("@triton.jit\ndef _warp_nearest_packed4(")
    end = body.find("\n\n@triton.jit", start + 1)
    if end < 0:
        end = len(body)
    part = body[start:end].replace("_warp_nearest_packed4(", "_warp_nearest_packed2_f32(", 1)
    part = part.replace("tl.uint32", "tl.uint64")
    a = part.index("    # The host admits")
    b = part.index("    _warp_nearest(", a)
    part = part[:a] + "    # Replicate-only, aligned float C4: two raw words preserve every pixel bit.\n    tl.static_assert(cn == 4 and BORDER == 1 and not TRANSPARENT)\n    border = 0\n" + part[b:]
    return part.replace("src_rs // 4, dst_rs // 4", "src_rs // 2, dst_rs // 2").replace("cn=1, BORDER=", "cn=2, BORDER=")


def derive(op):
    src = paths(op)[0]
    raw = open(src, "rb").read()
    text = raw.decode("utf-8")
    hits = text.count(ANCHOR)
    if hits != 1:
        sys.exit(f"{src}: the libdevice anchor appears {hits} times; the generator needs updating")
    header = (f"# Generated HIP {op} kernels; regenerate with tools/make_kernels_cpu_hip.py.\n"
              f"# source_sha256: {hashlib.sha256(raw).hexdigest()}\n")
    body = text.replace(ANCHOR, ANCHOR + shim_for(text), 1)
    for name in MISSING_ON_HIP:
        body = re.sub(rf"\blibdevice\.{name}\(", f"_triton_{name}(", body)
    left = re.findall(r"libdevice\.(?:add_rn|mul_rn|sub_rn|div_rn|rint|float2int_rn)", body)
    if left:
        sys.exit(f"unreplaced libdevice calls remain: {left[:3]}")
    if op == "resize":
        body = aligned_cubic_tail(body)
        body += "\n\n" + static_area_kernel(body)
        body += "\n\n" + cubic_row_reuse_kernel(body)
        body += "\n\n" + aligned_f32_cubic_kernel(body)
        fast = os.path.join(KERNELS, "resize_hip_fast.py")
        body += "\n\n# HIP-specific complete-window area kernels.\n" + open(fast, encoding="utf-8").read()
    elif op == "cvtcolor":
        body = compact_luv_lookup(body)
        body = reconstruct_luv_inverse(body)
        fast = os.path.join(KERNELS, "cvtcolor_hip_fast.py")
        body += "\n\n# HIP-specific aligned colour kernels.\n" + open(fast, encoding="utf-8").read()
    if op == "warpaffine":
        body += "\n\n# HIP channel-contiguous linear interpolation.\n" + warp_channel_kernel(body) + "\n\n" + warp_channel_kernel(body, "cubic")
        body += "\n\n" + warp_i32_channel_kernel(body)
        body += "\n\n" + packed_u8_cubic_kernel(body)
        for name in ("_warp_cubic", "_warp_cubic_i32_hip_channels", "_warp_cubic_packed4_u8"):
            body += "\n\n" + halfpixel_cubic_kernel(body, name)
        body += "\n\n" + packed_u16_nearest(body)
        body += "\n\n" + packed_f32_nearest(body)
    return header + body


def main():
    ops = [op for op in OPS if needs_derivation(op)]
    if "--check" in sys.argv[1:]:
        stale = []
        for op in ops:
            dst = paths(op)[1]
            have = open(dst, "rb").read() if os.path.exists(dst) else None
            if have != derive(op).encode("utf-8"):
                stale.append(op)
                print(f"kernels/{op}_cpu_hip.py is stale or missing; run tools/make_kernels_cpu_hip.py",
                      file=sys.stderr)
            else:
                print(f"kernels/{op}_cpu_hip.py is current")
        return 1 if stale else 0
    for op in ops:
        dst = paths(op)[1]
        with open(dst, "w", encoding="utf-8") as f:
            f.write(derive(op))
        print(f"wrote {os.path.normpath(dst)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
