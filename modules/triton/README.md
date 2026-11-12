# Triton-backed GPU operations

An optional opencv_contrib module whose image operators are written in
[Triton](https://github.com/triton-lang/triton) and compiled on first use at run
time. It provides `cv::triton::resize`, `cv::triton::cvtColor` and
`cv::triton::warpAffine`.

The module is **off by default** and does not modify any existing OpenCV
function. `cv::cuda::resize`, `cv::cuda::cvtColor` and `cv::cuda::warpAffine` are
untouched; a caller opts in by name by calling the `cv::triton::` entry.

## What it supports

Common: Linux, CUDA (driver API), `sm_90` tested; ROI source and destination;
the output is reused when it already has the right size and type. Unsupported
arguments raise `cv::Error::StsUnsupportedFormat`; invalid layouts, in-place
operations and overlapping src/dst are rejected as invalid arguments.
**There is no implicit fallback** to CPU or to `cv::cuda`: a call that returns
has submitted the computation.

### `cv::triton::resize`

| | |
|---|---|
| depth | `CV_8U`, `CV_32F` |
| channels | 1, 3, 4 |
| sizing | explicit `dsize`, or `fx`/`fy` when `dsize` is empty |
| routes | same size (a device copy), `INTER_NEAREST`, `INTER_LINEAR`, `INTER_AREA` (integer and fractional ratios), `INTER_CUBIC` |

A same-size copy does not perform interpolation, including when `INTER_CUBIC`
is requested. All four interpolation modes are served for both depths.

What it computes is OpenCV **CPU** `cv::resize` of a pinned reference
configuration, so uint8 output matches it bit-exactly. It is not a drop-in
replacement for `cv::cuda::resize`, which computes a different result.

### `cv::triton::cvtColor`

`CV_8U` only, and only the conversions OpenCV computes with **integer**
arithmetic — that is the set for which "bit-exact with the CPU" is a
well-defined promise:

| group | codes |
|---|---|
| channel permutes | `BGR2BGRA`, `BGRA2BGR`, `BGR2RGBA`, `RGBA2BGR`, `BGR2RGB`, `BGRA2RGBA` |
| to gray | `BGR2GRAY`, `RGB2GRAY`, `BGRA2GRAY`, `RGBA2GRAY` |
| YCrCb | `BGR2YCrCb`, `RGB2YCrCb`, `YCrCb2BGR`, `YCrCb2RGB` |
| YUV | `BGR2YUV`, `RGB2YUV`, `YUV2BGR`, `YUV2RGB` |
| HSV | `BGR2HSV`, `RGB2HSV`, `BGR2HSV_FULL`, `RGB2HSV_FULL` |

`HSV2BGR` and every HLS code are **refused on purpose**. OpenCV computes them
in float32, and in the build that was characterised the SIMD body and the scalar
tail round differently, so the result for a pixel depends on which of the two
processes it and therefore on where it sits in the row. The CPU result is
well-defined for a given build and a given pixel; what is missing is a value
that is stable across positions and builds to hold a GPU kernel to. Refusing
these codes by name is honest; serving them under a tolerance would not be.
Also refused: `CV_32F`, `CV_16U`, 2-channel images, and codes that change the
image size.

The reference here is the module's own `cv::cvtColor`. "Integer" is not what
makes that a promise — a different integer implementation is still different
numbers. What the integer arithmetic buys is that each output pixel depends only
on its own input bytes and on constant tables: no reduction order, no SIMD
width, and so the CPU result does not depend on where a pixel sits in the row.
That property was measured by a directed probe (64 colours at two widths), and
equality to the reference over all 256^3 colour triples for the 3-channel codes
— 452M pixels total; the 4-channel codes sample alpha rather than enumerate all
256^4. The probe is evidence for this build, not a proof for every build.

This is a statement about the build that was verified, not about "the CPU" in
general: `cv::cvtColor` dispatches through `cv_hal_cvtBGRtoBGR`, `cvtBGRtoYUV`
and `cvtBGRtoHSV`, so an IPP or custom-HAL build — or a later OpenCV with
different coefficients — is a different reference and needs its own
verification. The accuracy suite records `Intel IPP`, `Custom HAL` and the
run-time IPP state by value; when a build differs from the one the evidence came
from it reports that as a skipped diagnostic rather than a failure, because such
a build is a legitimate configuration. The numeric tests compare against the
portable C++ oracle compiled into the test binary, which no HAL can reroute, so a
build that computes different numbers still fails them.

### `cv::triton::warpAffine`

`CV_8U`, `CV_16U` and `CV_32F`; 1/3/4 channels; `INTER_NEAREST`, `INTER_LINEAR`
and `INTER_CUBIC`, with `INTER_AREA` as a LINEAR alias; the five standard border
modes plus `BORDER_TRANSPARENT` for NEAREST/LINEAR/AREA; and `WARP_INVERSE_MAP`
both ways.

| dimension | supported |
|---|---|
| depth | `CV_8U`, `CV_16U`, `CV_32F` |
| channels | 1, 3, 4 |
| interpolation | `INTER_NEAREST`, `INTER_LINEAR`, `INTER_CUBIC`; `INTER_AREA` aliases LINEAR |
| border modes | `CONSTANT`, `REPLICATE`, `REFLECT`, `REFLECT_101`, `WRAP`; `TRANSPARENT` for NEAREST/LINEAR/AREA |
| `borderValue` | a `Scalar` of up to four components, one per channel, kept in full precision |
| matrix | 2x3 `CV_32F` or `CV_64F`, forward map inverted internally unless `WARP_INVERSE_MAP` |

`INTER_LANCZOS4` is **refused on purpose**: its 8x8 weight table and clamped
taps are a separate contract, and approximating them behind an unmeasured
tolerance is not acceptable. Also refused: `CV_64F`, channels outside 1/3/4,
overlapping input/output, and `INTER_CUBIC` with `BORDER_TRANSPARENT`.

`INTER_AREA` follows LINEAR sampling here; it does not integrate a resize area.
`BORDER_TRANSPARENT` reads the previous destination. NEAREST preserves pixels
whose rounded source point is outside. LINEAR/AREA replace each out-of-range
tap with that destination pixel, retaining the original weights, and leave
wholly outside neighbourhoods untouched. Supply an initialized destination of
the requested size and type; allocating a new output supplies no defined
background. These routes have independent cases in `perf/perf_warpaffine.cpp`;
the native CUDA baseline does not establish matching transparent semantics.

For `INTER_NEAREST` and `INTER_LINEAR`, what it computes is OpenCV's CPU
`cv::warpAffine` on its **vector path**, and this needs stating because that
build is not self-consistent: its SIMD body evaluates a column's coordinate as
`v_fma(M0, x, (float)(y*M1 + M2))` (fused only on the selected FMA path) and its scalar tail as `x*M0 + y*M1 + M2`,
one ulp apart, and the tail's length depends on the destination width. Two
destination widths therefore disagree on the region they share — under
`INTER_NEAREST` by an adjacent pixel's value (measured up to 239 of 255 for
`CV_8U`), under `INTER_LINEAR` by at most one level in the measured range.
`CV_16U` behaves the same way. This module computes the vector-path result, so
it matches a natural-width CPU call except at those isolated pixels.

`INTER_CUBIC` is the exception, and it is measured rather than assumed:
`cv::warpAffine` routes it through `genericWarp` → `bicubicVec`, which evaluates
a column's coordinate as `M_x + M(0,0)*x` from the **absolute** x with no fma,
so there is no body/tail split. Over 80 seeds × 4 widths, 0 of 320 (seed, width)
pairs differ for `CV_8U`, `CV_16U` and `CV_32F`; its reference is therefore the
natural-width call.

The runtime calibrates fused versus split operations against the linked CPU reference
at rounding-sensitive coordinates; the result participates in the compile key.
Integer border components are rounded and saturated to their destination depth.
CUDA uses per-mode tiles: `1×256` for NEAREST and `2×64` for LINEAR/CUBIC
(including the AREA alias). HIP retains the measured launch-shape override with
small/narrow-image and C3 NEAREST fallbacks. These choices change launch geometry,
not sampling semantics.

**This is an open interface question, not a settled equivalence.** The module
asks the maintainers which behaviour a caller should be able to rely on; see
[the warpAffine guide](doc/warpaffine.markdown) for the reproduction and the
compatibility cost. Do not read this entry as "a drop-in match for
`cv::warpAffine` at any destination width".

## Building

```bash
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules \
      -DBUILD_opencv_triton=ON -DWITH_CUDA=ON \
      -DTRITON_PYTHON_EXECUTABLE=<python that can import triton> \
      -DTRITON_PYTHONPATH=<where triton is, if not on its default path> \
      <opencv>
```

An ordinary build needs no Python, Triton or PyTorch: with the module off, none
of them is involved. An enabled build never needs PyTorch. **Verified against
Triton 3.6.0 on CUDA 12.8**; the compile key carries the compiler version, so
artifacts from another Triton are not reused. The launcher currently accepts
only the validated Triton 3.6.0 ABI; other compiler versions are rejected.

A cold cache needs Python and Triton, which the module runs in a separate,
Torch-free helper process. The hot path — cache lookup and kernel launch — is
C++ and never enters an interpreter.

### HIP backend (Hygon BW1000 / gfx936, DTK)

```bash
ROCM_PATH=/opt/dtk cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules \
      -DBUILD_opencv_triton=ON -DWITH_CUDA=OFF -DTRITON_BACKEND=hip \
      -DTRITON_ROCM_PATH=/opt/dtk \
      -DTRITON_PYTHON_EXECUTABLE=<python that can import a hip Triton> \
      -DTRITON_PYTHONPATH=<flagtree, not upstream Triton> \
      <opencv>
```

`TRITON_BACKEND` is `auto` (CUDA when OpenCV has it, else HIP when a HIP toolkit is found),
`cuda` or `hip`. On a machine whose CUDA is a compatibility layer, say `hip` explicitly.
The HIP build runs the same runtime (`src/runtime/hip_runtime.cpp` is generated from
`cuda_runtime.cpp` by `tools/port_hip_runtime.py`) and the same kernels, with the libdevice
primitives flagtree's HIP build lacks shimmed (`tools/make_kernels_cpu_hip.py`); the configure
fails when either generated file is stale. `ROCM_PATH` must be set at run time too: the compile
helper's compiler finds its device libraries through it.

OpenCV has no `cv::cuda` on such a build, so the `GpuMat` entries raise
`StsNotImplemented`. Use the device-buffer entries in `opencv2/triton/device.hpp`
(`resizeDevice`, `cvtColorDevice`, `warpAffineDevice`): same validation, same runtime, same
results, with your own device buffers and a `hipStream_t`. The accuracy test is
`opencv_test_triton_hip` (`test/hip/`), against the same references and bounds as `test/`;
`test/`, `perf/` and `samples/` are `GpuMat`-based and are built on CUDA only.

### HIP examples and Python boundary

With `BUILD_EXAMPLES=ON`, `example_triton_device resize` and
`example_triton_device warp` demonstrate caller-owned HIP buffers, an explicit
stream, synchronization and a CPU comparison; neither needs Torch.

The Python module can be built and imported on HIP, and its backend queries
work. Its `resize` and `warpAffine` GpuMat adapters raise a catchable
`StsNotImplemented`; HIP numerical calls currently use the C++ device-buffer
entries. This change does not extend or modify cvtColor. The existing Python
cvtColor GpuMat adapter is outside this change and must not be advertised as a
HIP numerical entry.

## Using it

C++:

```cpp
#include <opencv2/triton.hpp>
#include <opencv2/imgcodecs.hpp>

cv::Mat host = cv::imread("in.png");
cv::cuda::GpuMat src(host), dst;

cv::cuda::Stream stream;
cv::triton::resize(src, dst, cv::Size(host.cols / 2, host.rows / 2), 0, 0,
                   cv::INTER_LINEAR, stream);
stream.waitForCompletion();   // asynchronous: wait before reading the result

cv::Mat out;
dst.download(out);
```

Python:

```python
import cv2 as cv
import numpy as np

img = np.random.randint(0, 256, (1080, 1920, 3), np.uint8)
gpu = cv.cuda_GpuMat(img)

dst = cv.triton.resize(gpu, (960, 540), interpolation=cv.INTER_LINEAR)
out = dst.download()          # download() synchronises

gray = cv.triton.cvtColor(gpu, cv.COLOR_BGR2GRAY).download()

# warpAffine: any 2x3 matrix; forward map, inverted internally.
M = cv.getRotationMatrix2D((img.shape[1] * 0.5, img.shape[0] * 0.5), 33.0, 0.8)
warped = cv.triton.warpAffine(gpu, M, (img.shape[1], img.shape[0])).download()
```

The C++ side is the same shape as the resize example:
`cv::triton::cvtColor(src, dst, cv::COLOR_BGR2YUV, stream)`.

The call is asynchronous on the stream you pass; inputs and output must stay
alive until the work completes.

## Samples

Three buildable examples, `samples/resize.cpp`, `samples/cvtcolor.cpp` and
`samples/warpaffine.cpp`; all are
built by `-DBUILD_EXAMPLES=ON` and need a Triton backend at run time:

```bash
./bin/example_triton_cvtcolor input.png gray.png     # BGR -> GRAY
./bin/example_triton_cvtcolor                        # synthetic input
./bin/example_triton_resize input.png out.png        # half size, INTER_LINEAR
```

## Further reading

- [resize guide](doc/resize.markdown) — resize in depth: dependencies, the JIT
  and its cache, the numerical contract, streams, limitations and troubleshooting.
  Everything except the resize-specific sections applies to `cvtColor` and
  `warpAffine` as well
- [warpAffine guide](doc/warpaffine.markdown) — which CPU path is the reference and
  why that question exists, the tolerance formulas, and the compatibility cost
- [API reference](include/opencv2/triton.hpp)
- Buildable samples: `samples/resize.cpp`, `samples/cvtcolor.cpp`,
  `samples/warpaffine.cpp`
- Tests: `test/test_resize.cpp`, `test/test_cvtcolor.cpp`, `test/test_warpaffine.cpp`
  (accuracy), `test/test_runtime.cpp` (lifetime and caching),
  `misc/python/test/test_triton.py` (binding and, on a reference build, the numbers)
- Benchmarks: `perf/perf_resize.cpp`, `perf/perf_cvtcolor.cpp`,
  `perf/perf_warpaffine.cpp` — each with the native `cv::cuda` path as a same-card
  timing baseline when the build has it (`perf/` only; the implementation never uses
  those modules)

## Status

Proposed for opencv_contrib; not yet merged. The public API and build switches
are part of the proposal and may change during review. The HIP backend has scoped verification on one
platform (Hygon BW1000, gfx936, DTK 26.04, flagtree 3.6.0); its routes are the CUDA backend's
(all four resize modes, and all three warpAffine modes).
