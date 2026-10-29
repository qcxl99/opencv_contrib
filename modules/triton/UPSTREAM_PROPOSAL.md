# `cv::triton`: a Triton-generated-kernel backend for `cv::cuda::resize`-shaped operators

Status: **proposal, not yet submitted**. This document exists to let OpenCV/
opencv_contrib maintainers evaluate the design and decide whether a PR is
worth pursuing, before one is opened. Nothing here is a claim that this is
ready to merge — it is a claim that it is ready to be *reviewed*.

## 1. What this is

A new, independent opencv_contrib module (`modules/triton`) that exposes
GPU-accelerated image operators whose kernels are written in
[Triton](https://github.com/triton-lang/triton) (MIT License) and compiled
**at OpenCV build time**, not at runtime. The first (and currently only)
operator is `resize`.

```cpp
#include <opencv2/triton.hpp>

cv::cuda::GpuMat src(h, w, CV_8UC3), dst;
cv::cuda::Stream stream;
cv::triton::resize(src, dst, cv::Size(w / 2, h / 2), 0, 0,
                    cv::INTER_LINEAR, stream);
stream.waitForCompletion();
```

```python
import cv2
dst = cv2.triton.resize(src, (w // 2, h // 2))
```

## 2. Design, and why it looks the way it does

**Separate namespace, not a `cv::cuda::resize` hook.** An earlier iteration of
this work patched `cv::cuda::resize` itself to dispatch into the Triton
kernel under an environment variable. That design is not what is proposed
here. The current design adds `cv::triton::resize` as its own entry point;
`cv::cuda::resize` is untouched, has zero diff from upstream in this fork,
and no existing caller's behavior can change by this module existing. A
caller opts in by name, explicitly.

**Why not "just put it in cv::cuda" instead of a new namespace?** Because
this module's supported subset is a strict and deliberately narrower slice
of `cv::cuda::resize`'s contract (see §3), and silently routing a subset of
calls to a different numeric implementation inside an existing, stable
function is a correctness and compatibility risk this design avoids
entirely. Growing the subset over time (§6) narrows the gap without ever
requiring that risk.

**Why compile Triton kernels at OpenCV build time rather than ship
prebuilt artifacts or JIT at runtime?** A build-time compile step keeps the
runtime dependency surface at zero — `WITH_TRITON=OFF` (the default) needs
no Python, Triton, or PyTorch at all, and most OpenCV users would build with
it off. With it on, `tools/compile_resize.py` produces versioned `.cubin`
artifacts plus a `kernels_manifest.json` (kernel source hash, Triton
version, target architecture, parameter ABI) under
`<build>/modules/triton/generated/`, so every artifact is traceable back to
the exact source that produced it.

**Why does a second backend (HIP/BW1000) already exist in this tree?** The
module's core validation and dispatch logic
(`src/core_resize.cpp`/`opencv2/triton/core.hpp`) is platform-agnostic by
construction — it operates on two POD structs (`ImageView`, `ResizeRequest`)
and never includes a CUDA or HIP header. `src/backend/{cuda,hip}/resize.cpp`
are the only platform-specific files, built mutually exclusively
(`TRITON_BACKEND_CUDA` vs `TRITON_BACKEND_HIP`). This was done so that a
second accelerator target is an additive backend file, not a rewrite of the
dispatch layer. **The HIP backend has not been validated on real BW1000
hardware** — it compiles and the artifact-loading path mirrors the CUDA one,
but this is explicitly unverified, not merely untested-by-omission.

## 3. Supported subset (first round)

| dimension | supported | rejected (raises `StsUnsupportedFormat`, never silently coerced) |
|---|---|---|
| depth | `CV_8U` | `CV_32F`, `CV_16U`, `CV_16S`, `CV_64F` |
| channels | 1, 3 | 2, 4 |
| interpolation | `INTER_NEAREST`, `INTER_LINEAR` | `INTER_AREA`, `INTER_CUBIC`, `INTER_LANCZOS4` |
| sizing | explicit `dsize` | `fx`/`fy` scale factors |
| layout | ROI src/dst (row-pitch addressing, not full-contiguity) | permuted/transposed views |
| batching | single image | N/A — no OpenCV type represents a batch of GpuMats |

Semantics match `cv::cuda::resize` **bit-exactly** for this subset (not
OpenCV CPU `resize` — the CUDA LINEAR rule has no half-pixel shift, so CPU
and CUDA disagree by design; the test suite checks against `cv::cuda`, not
CPU). There is no implicit fallback to CPU or to `cv::cuda`: a call that
returns without throwing has executed the Triton kernel.

## 4. Build and test integration

```bash
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules \
      -DWITH_TRITON=ON -DTRITON_BACKEND_CUDA=ON \
      -DPYTHON3_EXECUTABLE=<python with triton> <opencv>
```

- Follows the standard module macro chain: `ocv_add_module` →
  `ocv_set_module_sources` → `ocv_module_include_directories` →
  `ocv_create_module` → `ocv_add_accuracy_tests()` / `ocv_add_perf_tests()` /
  `ocv_add_samples()`.
- One deliberate deviation: source files are listed explicitly rather than
  via `ocv_glob_module_sources()`. Reason, documented in-file
  (`CMakeLists.txt` lines 21–27): the CUDA and HIP adapter files export the
  same three symbols and must never both be linked into one binary; a glob
  would pull in both backend subdirectories unconditionally.
- Tests are written against real OpenCV test infrastructure
  (`opencv2/ts.hpp`, `opencv2/ts/cuda_test.hpp`, `CUDA_TEST_P`,
  `CV_TEST_MAIN`), the same framework `cudawarping`'s own tests use — not a
  separate, project-specific test harness.
- Verified this round: `ninja opencv_triton`, `ninja opencv_python3`, and
  `ninja opencv_test_triton` all build clean; the full test suite passes
  **975/975**.

## 5. Coding-style and contribution-process conformance

Checked directly against [How_to_contribute](https://github.com/opencv/opencv/wiki/How_to_contribute)
and [Coding_Style_Guide](https://github.com/opencv/opencv/wiki/Coding_Style_Guide),
file by file, not from memory. Items found non-conformant were fixed in this
round (uncommitted in this fork at time of writing) and re-verified by
rebuilding and re-running the full test suite after each fix.

| requirement | finding | resolution |
|---|---|---|
| Public functions intended for Python/Java must be marked `CV_EXPORTS_W`, not plain `CV_EXPORTS` | `resize()` was `CV_EXPORTS` only. Verified mechanism: the Python binding generator (`hdr_parser.py`) does a literal string match on the token `CV_EXPORTS_W`; `cvdef.h` defines `CV_EXPORTS_W` as `CV_EXPORTS`, so the two are behaviorally identical to the compiler but the generator only sees one of them. **Net effect before the fix: `cv2.triton.resize(...)` did not exist**, while `cv2.triton.haveBackend()`/`backendName()` (both already `CV_EXPORTS_W`) did. Precedent check: `cv::cuda::resize`, the function `cv::triton::resize` is explicitly modeled on, is `CV_EXPORTS_W` in `cudawarping.hpp`. | Fixed. Rebuilt `opencv_python3`; confirmed `cv2.triton.resize` is now a real callable. |
| Every source file must open with the standard OpenCV license header | None of the 16 `.hpp`/`.cpp` files had it. | Fixed in the 15 files the guide requires it in (`samples/resize.cpp` is explicitly exempt). Header text matched verbatim against `modules/cannops`, a module that already carries it correctly. |
| `.cpp` files must include `precomp.hpp` first, for the MSVC precompiled-header mechanism | `src/core_resize.cpp` included its headers directly instead. | Fixed — confirmed `precomp.hpp` already guards its CUDA/HIP-specific includes behind `OPENCV_TRITON_HAVE_CUDA_ADAPTER`, so switching does not compromise this file's platform-agnostic design. |
| Test naming: `TEST(<module_name>_<tested_class_or_function>, <type>)` | Fixtures were named `TritonResize`/`TritonResizeInterop` (no module-name-underscore prefix). | Renamed to `Triton_Resize`/`Triton_ResizeInterop` throughout `test_resize.cpp`. |
| `modules/README.md` lists every contrib module | `triton` was absent. | Added, alphabetically positioned between `text` and `tracking`. |
| Each module should have its own `README.md` | Absent. | **Not yet fixed** — see §7, item 4, and the precedent check below before treating this as a clean gap. |
| Header guard format `OPENCV_module_name_header_name_HPP` | Initially suspected non-conformant (`OPENCV_TRITON_HPP` looked one segment short). Checked against `cudawarping.hpp`'s own guard (`OPENCV_CUDAWARPING_HPP` — also one segment, because the module name and the umbrella header's file name coincide): same pattern, already conformant. | No change needed; this was a false positive caught before editing. |
| No GPL/LGPL or research-only licensed code | N/A for this module's own code (original implementation). Triton itself is MIT-licensed, compatible with OpenCV's license, and is a **build-time-only** dependency — not linked into the runtime binary when `WITH_TRITON=OFF`. | Confirmed, no action needed. |

**On the module-README gap specifically**: before writing one, the
precedent was checked rather than assumed. All 11 `cuda*` modules in this
contrib tree (`cudaarithm`, `cudabgsegm`, `cudacodec`, `cudafeatures2d`,
`cudafilters`, `cudaimgproc`, `cudalegacy`, `cudaobjdetect`, `cudaoptflow`,
`cudastereo`, `cudawarping`) lack a module-level `README.md` — 0 of 11.
Repo-wide, 21 of 63 modules lack one. The written requirement exists and is
real, but it is observed at roughly the same (low) rate across the module
family this one most resembles. Writing one anyway is reasonable on its own
merits for a module this architecturally unusual (two backends, a
build-time compiler dependency) — but the justification is "useful given
what's unusual here," not "every comparable module does this and we
missed it." Content is deferred until the open design questions in §7 are
resolved, since a README written before that would need rewriting.

## 6. What is implemented upstream of this module but not yet wired in here

The underlying Triton kernel (outside this OpenCV fork, in the originating
project) already supports AREA/CUBIC interpolation, `float32`/`CV_8U` C4,
`fx`/`fy` scale factors, and a batched calling convention. None of that is
currently reachable through `cv::triton::resize`. Per-item assessment:

| capability | blocker inside this module | relative cost |
|---|---|---|
| `INTER_CUBIC` | None structural — shares the same parameter ABI as LINEAR/NEAREST in the underlying kernel. Needs a mode branch in `core_resize.cpp`'s validation and `backend/cuda/resize.cpp`'s artifact-name lookup (currently hardcoded to `"linear"`/`"nearest"` only). | Low |
| 4 channels | `core_resize.cpp` rejects outright. The vendored `kernels/resize.py` is an older snapshot without C4 support and needs re-vendoring from the current upstream kernel; `tools/compile_resize.py`'s channel loop needs a third value added. | Medium |
| `fx`/`fy` | `cuda_resize_adapter.cpp` rejects any non-zero value. The `fx`/`fy` fields that do exist in `ResizeRequest` are derived *from* an already-resolved `dsize`, not used to derive one — there is currently no `cvRound`/`saturate_cast<int>` logic in this module to reuse; it would need to be ported in from scratch, matching `cv::cuda::resize`'s own parameter-resolution rule bit-for-bit. | Medium |
| `float32` | `core_resize.cpp` rejects outright. **Not a simple unlock**: `variantName()` — the function that picks which compiled artifact to load — keys only on `{mode, channels, 4 alignment bits}`, with no dtype dimension at all. Removing the depth check without adding a dtype axis to the key would make a `float32` call compute the same artifact name as the `uint8` call and silently load the wrong cubin. Requires a new key dimension and a doubled artifact set. | Medium-high, and the only item on this list with a silent-wrong-result risk if done incompletely |
| `INTER_AREA` | Not implemented at all in this module (no validation branch, no artifact-name support). The upstream kernel's AREA path depends on two runtime-resolved constexprs (tap count, integer-vs-fractional ratio) that make its ahead-of-time compiled artifact space far larger than every other mode's (roughly 672 artifacts vs. ~48) — a cost/benefit tradeoff the upstream kernel project resolved by excluding AREA from its own AOT delivery path entirely. This module would need to either adopt the same exclusion or find a different answer; either way it is a design decision, not an implementation task. | High, architectural |
| batch | No batch concept anywhere in this module's request types, and `cv::cuda::GpuMat` itself has no batch representation to adapt from (confirmed: `cudawarping.hpp` has exactly one single-image `resize` overload). Needs a new public API shape designed before any code is written. | High, architectural |

## 7. Open questions for maintainers

These are the items that genuinely need a maintainer's input, not just more
engineering:

1. **Target branch.** No submission branch has been chosen yet. The
   contribution guide requires choosing the right base branch explicitly;
   this has not been done.
2. **AREA's AOT cost tradeoff.** Is the upstream project's own exclusion of
   `INTER_AREA` from ahead-of-time compilation (due to its larger artifact
   space) an acceptable precedent for this module too, or should it be
   solved differently here?
3. **Batch API shape.** If batched resize is wanted in this module, what
   should the public type look like, given `GpuMat` has no batch
   representation to start from?
4. **Module README content**, once 2 and 3 have answers (writing it before
   would mean rewriting it after).
5. **Scope and pacing.** Should CUBIC/C4/`fx`/`fy`/`float32`/AREA/batch be
   completed before a first PR, or is a narrower first submission (current
   subset plus the compliance fixes in §5) preferred, with the rest as
   follow-ups?

## 8. Verification record for this round's changes

- `ninja opencv_triton` — clean build.
- `ninja opencv_python3` — clean build; `cv2.triton.resize` and
  `cv2.triton.haveBackend` confirmed present and callable from a live
  Python interpreter against the rebuilt binding.
- `ninja opencv_test_triton` — clean build; `./bin/opencv_test_triton` run
  in full (no filter): **975/975 tests passed**, matching the test count
  recorded in this fork's own history after the ROI-support merge, so this
  round's renames and `precomp.hpp` ordering fix did not change test
  behavior or coverage, only naming and include order.
- All changes described in §5 are currently **uncommitted** in the
  `kernelforge/triton-resize` branch of this `opencv_contrib` fork.
