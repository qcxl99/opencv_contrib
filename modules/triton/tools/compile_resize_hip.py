#!/usr/bin/env python3
"""Build-time AOT compilation of the Triton resize kernels -- HIP/hcu backend.

Sibling of compile_resize.py (CUDA). Same specialization space and the same
manifest shape, with two platform differences recorded explicitly rather than
reused silently:

  - emits .hsaco artifacts (k.asm["hsaco"]), not .cubin
  - records warp_size from the kernel's own compiled metadata instead of
    assuming 32: BW1000 (gfx936) warps are 64 lanes wide, and the C++ launcher
    needs threads-per-block = num_warps * warp_size to be correct on either
    platform rather than hard-coding the CUDA value.

This file must run under a Python/Triton pair that actually targets hcu
(flagtree's triton, not upstream) -- see bw1000-triton-cv-env memory. It does
not select that environment itself; the caller (CMake) is responsible for
invoking it with the right PYTHONPATH/interpreter.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import sys

BLOCK_H, BLOCK_W, NUM_WARPS = 1, 256, 4


def load_kernels(path):
    spec = importlib.util.spec_from_file_location("kf_triton_kernels_hip", path)
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

    entries = {}
    warp_size = None
    gcn_arch = None
    for mode, kern in (("linear", mod._resize_linear_kernel),
                       ("nearest", mod._resize_nearest_kernel)):
        for ch in (1, 3):
            for sp, dp, ss, ds in itertools.product((1, 0), repeat=4):
                # Shape the buffers so the requested divisibility holds.
                srow = 64 * ch + (0 if ss else 1)
                drow = 32 * ch + (0 if ds else 1)
                sbuf = torch.zeros((64, srow + 16), dtype=torch.uint8, device="cuda")
                dbuf = torch.zeros((32, drow + 16), dtype=torch.uint8, device="cuda")
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

                k = kern.warmup(s, d, 64, 64, 32, 32, s.stride(0), d.stride(0), ch,
                                2.0, 2.0, grid=(1, 1),
                                BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, num_warps=NUM_WARPS)
                k._init_handles()
                if warp_size is None:
                    warp_size = k.metadata.warp_size
                    gcn_arch = k.metadata.arch
                variant = f"{mode}_c{ch}_sp{sp}dp{dp}ss{ss}ds{ds}"
                with open(os.path.join(args.out_dir, variant + ".hsaco"), "wb") as fh:
                    fh.write(k.asm["hsaco"])
                entries[variant] = {
                    "function": k.name,
                    "mode": mode,
                    "channels": ch,
                    "src_ptr_div16": bool(sp),
                    "dst_ptr_div16": bool(dp),
                    "src_stride_div16": bool(ss),
                    "dst_stride_div16": bool(ds),
                    "block_h": BLOCK_H,
                    "block_w": BLOCK_W,
                    "num_warps": NUM_WARPS,
                    "warp_size": k.metadata.warp_size,
                    "shared_bytes": k.metadata.shared,
                    "n_regs": k.n_regs,
                    "artifact_hash": k.metadata.hash,
                }

    manifest = {
        "kernel_source": os.path.basename(args.kernel),
        "kernel_source_sha256": src_hash,
        "triton_version": triton.__version__,
        "triton_distribution": "flagtree",
        "target_gcn_arch": gcn_arch,
        "device_name": torch.cuda.get_device_name(0),
        "warp_size": warp_size,
        "param_abi": "ptr,ptr,u32x6,f32x2,ptr,ptr",
        "runtime_arguments": ["src_h", "src_w", "dst_h", "dst_w",
                              "src_step", "dst_step", "fx", "fy"],
        "variants": entries,
        "variant_count": len(entries),
        "numerical_note": ("no PTX-equivalent instruction pin on this backend; "
                           "validate against OpenCV CPU under the compatibility "
                           "criterion (atol<=1), not atol=0"),
    }
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"triton(hip): generated {len(entries)} variants -> {args.out_dir}")
    return 0 if entries else 1


if __name__ == "__main__":
    sys.exit(main())
