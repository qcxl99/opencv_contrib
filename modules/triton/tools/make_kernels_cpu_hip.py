#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

"""Generate HIP kernel sources from the module's CUDA-compatible sources.

Run without arguments to write kernels/<op>_cpu_hip.py, or use --check to verify
that the checked-in files match the current sources and this generator.

The source kernels use libdevice operations to preserve floating-point rounding.
The HIP libdevice implementation does not provide every operation, so this tool
substitutes the shims below. Kernels without those imports can be used directly.
Resize also includes the module's HIP-specific complete-window area kernels.

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


# --- HIP/AMDGCN shims for the libdevice functions this file needs that
# --- flagtree's hip libdevice does not provide (add_rn / mul_rn / div_rn).
# --- Semantics preserved: each is a SINGLE round-to-nearest operation that the
# --- compiler may not reassociate or fuse into a neighbouring term, which is
# --- exactly why the CPU-aligned kernels spell them out instead of using `+`
# --- and `*`. v_add_f32 / v_mul_f32 are the CDNA instructions for add.f32 /
# --- mul.f32; CDNA's rounding is round-to-nearest-even unconditionally.
# --- `div_rn` has no single instruction on CDNA (the divide is a 5-instruction
# --- sequence); plain `/` is used.
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
MISSING_ON_HIP = ("add_rn", "mul_rn", "div_rn", "sub_rn")
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
    arith = [n for n in used if n != "div_rn"]
    isa = (" ".join(f"v_{n.split('_')[0]}_f32" for n in arith) + " are the CDNA instructions for "
           + " / ".join(f"{n.split('_')[0]}.f32" for n in arith)) if arith else ""
    head = ("\n\n# --- HIP/AMDGCN shims for the libdevice functions this file needs that\n"
            "# --- flagtree's hip libdevice does not provide (" + " / ".join(used) + ").\n"
            "# --- Semantics preserved: each is a SINGLE round-to-nearest operation that the\n"
            "# --- compiler may not reassociate or fuse into a neighbouring term.\n")
    if isa:
        head += ("# --- " + isa + "; CDNA rounds to nearest-even\n"
                 "# --- unconditionally.\n")
    if "div_rn" in used:
        head += ("# --- `div_rn` has no single instruction on CDNA (the divide is a 5-instruction\n"
                 "# --- sequence); plain `/` is used.\n")
    return head + "".join(_SHIMS[n] for n in used)


def paths(op):
    return (os.path.join(KERNELS, f"{op}_cpu.py"), os.path.join(KERNELS, f"{op}_cpu_hip.py"))


def needs_derivation(op):
    src = paths(op)[0]
    return os.path.isfile(src) and ANCHOR in open(src, encoding="utf-8").read()


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
    left = re.findall(r"libdevice\.(?:add_rn|mul_rn|sub_rn|div_rn)", body)
    if left:
        sys.exit(f"unreplaced libdevice calls remain: {left[:3]}")
    if op == "resize":
        fast = os.path.join(KERNELS, "resize_hip_fast.py")
        body += "\n\n# HIP-specific complete-window area kernels.\n" + open(fast, encoding="utf-8").read()
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
