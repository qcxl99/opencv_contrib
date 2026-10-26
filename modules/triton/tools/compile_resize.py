#!/usr/bin/env python3
"""Build-time AOT compilation of the Triton resize kernels.

Emits one cubin per runtime specialization plus a manifest recording the
source hash, compiler, target and parameter ABI, so a built artifact can be
traced back to its inputs and rebuilt.

Specialization space, measured rather than assumed: Triton specializes on
divisibility-by-16 of both the pointer and the byte stride, and `channels` is
a tl.constexpr. That is 2 modes x 2 channel counts x 2^4 divisibility
combinations = 64 variants. Keying on stride alone caused misaligned accesses
during development.

Image height/width and strides stay RUNTIME arguments, so no artifact is
generated per test size.
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

    entries = {}
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
                variant = f"{mode}_c{ch}_sp{sp}dp{dp}ss{ss}ds{ds}"
                with open(os.path.join(args.out_dir, variant + ".cubin"), "wb") as fh:
                    fh.write(k.asm["cubin"])
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
        "variants": entries,
        "variant_count": len(entries),
    }
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"triton: generated {len(entries)} variants -> {args.out_dir}")
    return 0 if entries else 1


if __name__ == "__main__":
    sys.exit(main())
