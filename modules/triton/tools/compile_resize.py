#!/usr/bin/env python3
"""Build-time AOT compilation of the Triton resize kernels.

Emits one cubin per runtime specialization plus a manifest recording the
source hash, compiler, target and parameter ABI, so a built artifact can be
traced back to its inputs and rebuilt.

Specialization space, measured rather than assumed: Triton specializes on
divisibility-by-16 of both the pointer and the byte stride, `channels` is a
tl.constexpr, and -- checked directly, NOT inferred from the kernel source
having no OUT_FLOAT32 branch -- on the ACTUAL dtype of every tensor passed
to warmup(), because tl.load/tl.store's element width comes from the
pointed-to tensor's dtype at trace time, not a runtime argument. A kernel
with no explicit dtype constexpr (NEAREST) is NOT dtype-transparent at the
compiled-artifact level for this reason: compiling it once against a uint8
probe buffer and reusing that exact cubin for a float32 call was tried and
is wrong (an 8360-byte u8 cubin vs an 8232-byte f32 cubin, different
machine code, confirmed by hashing k.asm['cubin'] for both) -- it reads/
writes the wrong element width and produces garbage. So every mode gets its
own compile pass per dtype with no sharing: 3 modes x 2 dtypes x 3 channel
counts x 2^4 divisibility combinations = 288 variants. Keying on stride
alone caused misaligned accesses during development.

Image height/width and strides stay RUNTIME arguments, so no artifact is
generated per test size -- but Triton's AOT specializer does not treat
plain int/float arguments as free of compile-time assumptions just because
the kernel signature has no tl.constexpr for them: it inspects the ACTUAL
VALUE passed to warmup() and, if that value happens to be divisible by 16
(same test it applies to pointers/strides), bakes in a "this argument is a
multiple of 16" assumption that then applies to every call against that
cubin. Probing with src_h=64/src_w=64/dst_h=32/dst_w=32 (all multiples of
16) silently baked that assumption onto src_h/src_w/dst_h/dst_w themselves
-- confirmed by hashing: those exact warmup() calls with src_h=64 vs
src_h=1 produce DIFFERENT cubins (16 vs 14 registers) even though src_h is
a plain runtime i32 with no constexpr anywhere in the kernel source, and
the src_h=64 cubin gives wrong (all-zero, as if every row but the first
were masked out) output for an actual runtime src_h=1 call -- reproduced
with a 1x64 CV_32FC1 source resized with INTER_NEAREST, ONE LINE beyond
what any prior test case in this module exercised, which is why this
pre-existing bug (present for the original uint8-only LINEAR/NEAREST build
too, not introduced by CUBIC/C4/f32/fx-fy) had not surfaced yet. The probe
integers below are therefore deliberately NOT multiples of 16 (65/63
instead of 64/64, 33/31 instead of 32/32), so the compiled cubin carries no
assumption about src_h/src_w/dst_h/dst_w beyond what the kernel source
itself declares -- ptr/stride divisibility is still probed exactly as
before, since those ARE meant to be compile-time specialized.

AREA is keyed by one MORE axis than LINEAR/NEAREST/CUBIC: INTEGER_RATIO
(IntegerAreaFilter vs AreaFilter are different compiled FORMULAS, selected
at compile time -- see kernels/resize.py's _resize_area_kernel docstring),
doubling AREA's own variant count relative to the other three modes but
NOT multiplying the whole specialization space by it -- tap count
(max_taps_x/y) is a plain runtime i32 argument, not a constexpr, and is
therefore NOT a key axis at all (ported from kernelforge-lab's own
bridge/triton_resize_launcher.cpp area_tap_bucket()/
kf_launcher_resolve_area_key(), the reference this module's AREA key
format matches). AREA also takes scale_x/scale_y (f32) where the other
three modes take fx/fy -- same quantity, AREA's own kernel parameter
names -- and two extra trailing runtime args (max_taps_x, max_taps_y)
inserted before the scratch pointers.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import sys

BLOCK_H, BLOCK_W, NUM_WARPS = 1, 256, 4

# dtype axis: (manifest key, torch dtype, kernel kwarg for OUT_FLOAT32 or
# None when the kernel takes no such constexpr at all).
DTYPES = (("u8", "uint8"), ("f32", "float32"))


def load_kernels(path):
    spec = importlib.util.spec_from_file_location("kf_triton_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)

    # torch is used only to obtain device buffers for warmup-compilation at
    # BUILD time. It is not a runtime dependency of the OpenCV module.
    import torch
    import triton

    mod = load_kernels(args.kernel)
    src_hash = hashlib.sha256(open(args.kernel, "rb").read()).hexdigest()

    # (mode, kernel, has_out_float32): NEAREST takes no OUT_FLOAT32 kwarg --
    # passing one would be a TypeError -- because its kernel body never
    # branches on dtype (plain tl.load/tl.store, no round/saturate step).
    # That does NOT make its two dtype cubins interchangeable; see the
    # module docstring. Both are compiled below regardless. AREA is NOT
    # in this list -- its parameter list and extra INTEGER_RATIO axis
    # differ enough from the other three that it gets its own loop below
    # rather than a branch inside this one.
    MODES = (
        ("linear", mod._resize_linear_kernel, True),
        ("nearest", mod._resize_nearest_kernel, False),
        ("cubic", mod._resize_cubic_kernel, True),
    )

    entries = {}
    for mode, kern, has_f32_kwarg in MODES:
        for dtype_key, dtype_name in DTYPES:
            torch_dtype = getattr(torch, dtype_name)
            for ch in (1, 3, 4):
                for sp, dp, ss, ds in itertools.product((1, 0), repeat=4):
                    # Shape the buffers so the requested divisibility holds.
                    srow = 64 * ch + (0 if ss else 1)
                    drow = 32 * ch + (0 if ds else 1)
                    sbuf = torch.zeros((64, srow + 16), dtype=torch_dtype, device="cuda")
                    dbuf = torch.zeros((32, drow + 16), dtype=torch_dtype, device="cuda")
                    soff = 0 if sp else 1
                    doff = 0 if dp else 1
                    s = sbuf[:, soff:soff + srow]
                    d = dbuf[:, doff:doff + drow]
                    if (s.data_ptr() % 16 == 0) != bool(sp):
                        continue
                    if (d.data_ptr() % 16 == 0) != bool(dp):
                        continue
                    if (s.stride(0) % 16 == 0) != bool(ss):
                        continue
                    if (d.stride(0) % 16 == 0) != bool(ds):
                        continue

                    kwargs = {"BLOCK_H": BLOCK_H, "BLOCK_W": BLOCK_W, "num_warps": NUM_WARPS}
                    if has_f32_kwarg:
                        kwargs["OUT_FLOAT32"] = (dtype_key == "f32")
                    if mode == "nearest":
                        # WORD_ALIGNED (default True) reinterprets src_ptr
                        # as uint32 and bit-shifts out 3 BYTES per pixel --
                        # a uint8 memory-layout trick (1 byte/channel) that
                        # reads garbage for float32 (4 bytes/channel). Left
                        # at its default here, the f32 C3 nearest cubin
                        # silently baked in the uint8 bit-shift path and
                        # produced wrong output for every real float32 C3
                        # NEAREST call -- caught by the test matrix, not
                        # anticipated. The runtime dispatch in this same
                        # file's resize() already routes float32 C3 away
                        # from this path at the Python-eager layer; this is
                        # the AOT-compile-time half of that same guard.
                        kwargs["WORD_ALIGNED"] = (dtype_key == "u8")
                    # src_h/src_w/dst_h/dst_w/fx/fy are placeholder VALUES
                    # for a pure trace-and-compile call (warmup() never
                    # executes the kernel), so they need not match s/d's
                    # real shape -- but they must NOT be multiples of 16
                    # (or 0 or 1), or Triton's AOT specializer bakes that
                    # in as if it were a declared constexpr. See the module
                    # docstring for the exact failure this caused.
                    k = kern.warmup(s, d, 65, 63, 33, 31, s.stride(0), d.stride(0), ch,
                                    2.3, 1.7, grid=(1, 1), **kwargs)
                    k._init_handles()
                    variant = f"{mode}_{dtype_key}_c{ch}_sp{sp}dp{dp}ss{ss}ds{ds}"
                    with open(os.path.join(args.out_dir, variant + ".cubin"), "wb") as fh:
                        fh.write(k.asm["cubin"])
                    entry = {
                        "function": k.name,
                        "mode": mode,
                        "dtype": dtype_key,
                        "channels": ch,
                        "src_ptr_div16": bool(sp),
                        "dst_ptr_div16": bool(dp),
                        "src_stride_div16": bool(ss),
                        "dst_stride_div16": bool(ds),
                        "block_h": BLOCK_H,
                        "block_w": BLOCK_W,
                        "num_warps": NUM_WARPS,
                        "shared_bytes": k.metadata.shared,
                        "n_regs": k.n_regs,
                        "artifact_hash": k.metadata.hash,
                    }
                    entries[variant] = entry

    # AREA: same dtype x channels x 4-alignment-bit sweep as the MODES loop
    # above, plus INTEGER_RATIO (2 values) as a fifth key axis. max_taps_x/y
    # are plain runtime args for warmup() purposes -- any value works, they
    # are not probed or baked in, so 2/2 is as good as any other choice (see
    # module docstring: tap count is not a key axis at all).
    AREA_BLOCK_H, AREA_BLOCK_W, AREA_NUM_WARPS = 1, 128, 2
    for dtype_key, dtype_name in DTYPES:
        torch_dtype = getattr(torch, dtype_name)
        for ch in (1, 3, 4):
            for integer_ratio in (True, False):
                for sp, dp, ss, ds in itertools.product((1, 0), repeat=4):
                    srow = 64 * ch + (0 if ss else 1)
                    drow = 32 * ch + (0 if ds else 1)
                    sbuf = torch.zeros((64, srow + 16), dtype=torch_dtype, device="cuda")
                    dbuf = torch.zeros((32, drow + 16), dtype=torch_dtype, device="cuda")
                    soff = 0 if sp else 1
                    doff = 0 if dp else 1
                    s = sbuf[:, soff:soff + srow]
                    d = dbuf[:, doff:doff + drow]
                    if (s.data_ptr() % 16 == 0) != bool(sp):
                        continue
                    if (d.data_ptr() % 16 == 0) != bool(dp):
                        continue
                    if (s.stride(0) % 16 == 0) != bool(ss):
                        continue
                    if (d.stride(0) % 16 == 0) != bool(ds):
                        continue

                    kwargs = {
                        "BLOCK_H": AREA_BLOCK_H, "BLOCK_W": AREA_BLOCK_W,
                        "num_warps": AREA_NUM_WARPS,
                        "INTEGER_RATIO": integer_ratio,
                        "OUT_FLOAT32": (dtype_key == "f32"),
                    }
                    # scale_x/scale_y (f32, not fx/fy -- AREA's own kernel
                    # parameter names) and max_taps_x/max_taps_y (plain
                    # runtime i32, not probed): 2.3/1.7/2/2 are placeholder
                    # VALUES, same role as the 65/63/33/31 dims above --
                    # not multiples of 16, so nothing is baked in.
                    k = mod._resize_area_kernel.warmup(
                        s, d, 65, 63, 33, 31, s.stride(0), d.stride(0), ch,
                        2.3, 1.7, 2, 2, grid=(1, 1), **kwargs)
                    k._init_handles()
                    variant = (f"area_{dtype_key}_c{ch}_ir{int(integer_ratio)}_"
                               f"sp{sp}dp{dp}ss{ss}ds{ds}")
                    with open(os.path.join(args.out_dir, variant + ".cubin"), "wb") as fh:
                        fh.write(k.asm["cubin"])
                    entries[variant] = {
                        "function": k.name,
                        "mode": "area",
                        "dtype": dtype_key,
                        "channels": ch,
                        "integer_ratio": integer_ratio,
                        "src_ptr_div16": bool(sp),
                        "dst_ptr_div16": bool(dp),
                        "src_stride_div16": bool(ss),
                        "dst_stride_div16": bool(ds),
                        "block_h": AREA_BLOCK_H,
                        "block_w": AREA_BLOCK_W,
                        "num_warps": AREA_NUM_WARPS,
                        "shared_bytes": k.metadata.shared,
                        "n_regs": k.n_regs,
                        "artifact_hash": k.metadata.hash,
                    }

    manifest = {
        "kernel_source": os.path.basename(args.kernel),
        "kernel_source_sha256": src_hash,
        "triton_version": triton.__version__,
        "target_arch": f"sm_{torch.cuda.get_device_capability()[0]}"
                       f"{torch.cuda.get_device_capability()[1]}",
        "device_name": torch.cuda.get_device_name(0),
        "param_abi": "ptr,ptr,u32x6,f32x2,ptr,ptr",
        "runtime_arguments": ["src_h", "src_w", "dst_h", "dst_w",
                              "src_step", "dst_step", "fx", "fy"],
        # AREA's ABI differs from the one above: scale_x/scale_y in place of
        # fx/fy (same quantity, AREA's own kernel parameter names) plus two
        # extra trailing runtime args (max_taps_x, max_taps_y) before the
        # scratch pointers. Each AREA variant entry also carries
        # "integer_ratio", the fifth key axis the other three modes don't
        # have.
        "param_abi_area": "ptr,ptr,u32x6,f32x2,u32x2,ptr,ptr",
        "runtime_arguments_area": ["src_h", "src_w", "dst_h", "dst_w",
                                   "src_step", "dst_step",
                                   "scale_x", "scale_y",
                                   "max_taps_x", "max_taps_y"],
        "variants": entries,
        "variant_count": len(entries),
    }
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"triton: generated {len(entries)} variants -> {args.out_dir}")
    return 0 if entries else 1


if __name__ == "__main__":
    sys.exit(main())
