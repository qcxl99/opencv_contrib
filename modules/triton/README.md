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
| routes | same size (a device copy), `INTER_NEAREST`, `INTER_LINEAR`, `INTER_CUBIC`, `INTER_AREA` (integer and fractional ratios) |

A same-size copy does not perform interpolation, including when `INTER_CUBIC`
is requested. Nonidentity CUBIC and fractional AREA execute their generic CPU-aligned kernels.

What it computes is OpenCV **CPU** `cv::resize` of a pinned reference
configuration, so uint8 output matches it bit-exactly. It is not a drop-in
replacement for `cv::cuda::resize`, which computes a different result.

### `cv::triton::cvtColor`

The native module serves **84 distinct numeric conversion codes** (enum aliases
share a code). It is wider than the KernelForge Python experiment adapter, which
serves four uint8 codes on contiguous tensors. Native support is:

| group | source → destination channels | depths |
|---|---|---|
| six channel permutes (`BGR2BGRA`, `BGRA2BGR`, `BGR2RGBA`, `RGBA2BGR`, `BGR2RGB`, `BGRA2RGBA`) | 3/4 → 3/4 | 8U, 32F |
| colour to gray (`BGR`, `RGB`, `BGRA`, `RGBA`) | 3/4 → 1 | 8U, 16U, 32F |
| gray to colour (`GRAY2BGR`, `GRAY2BGRA`) | 1 → 3/4 | 8U, 16U, 32F |
| YCrCb and YUV, both BGR/RGB directions | 3 → 3 | 8U, 16U, 32F |
| forward HSV and HSV_FULL, BGR/RGB | 3 → 3 | 8U |
| packed BGR565/BGR555, colour/gray encode and decode | 1/3/4 ↔ 2 | 8U |
| Bayer bilinear: BG/GB/RG/GR to BGR, BG to BGRA/GRAY | 1 → 1/3/4 | 8U, 16U |
| XYZ (four), Lab (eight), Luv (six; no forward linear Luv) | 3 → 3 | 8U |
| UYVY/YUY2/YVYU decode to BGR/RGB (six) | 2 → 3 | 8U; even width |
| `YUV2GRAY_UYVY`, `YUV2GRAY_YUY2` channel extraction | 2 → 1 | 8U; odd width allowed |
| NV12/NV21 decode to BGR/RGB/BGRA/RGBA (eight) | 1 → 3/4 | 8U; packed source `(3H/2) × W`, even W, source rows divisible by 3 |

The code fixes the channel count; there is no `dstCn` override. Packed 565/555
uses **CV_8UC2**, two bytes per pixel, rather than CV_16UC1. There are 20 numeric
codes at 16U and 20 at 32F. Except for NV12/NV21 decode, output size equals input.
ROI and row pitch are supported for interleaved GpuMat pixels. An existing output
is reused when its size/type match; otherwise OpenCV OutputArray creates a fitting
buffer. Source/output overlap is refused. Work is asynchronous on the supplied
stream; callers order producers/consumers and retain image allocations until it
completes. Internal compiled modules and tables have their own in-flight lifetime.

8U/16U conversions and 32F channel copies compare exactly with this build's own
CPU `cv::cvtColor`. Finite 32F arithmetic (colour to gray, YCrCb and YUV) uses:

```
|got - ref| <= 8 * 2^-24 * max(1, |ref|, max(abs(input channels at this pixel)))
```

The scale is per pixel, not the maximum of the whole image. Added alpha is 255,
65535 or 1.0 at 8U, 16U or 32F. Float arithmetic is not a bit-exact promise:
CPU SIMD bodies, scalar tails and FMA configurations can evaluate different
expressions. Numerical comparisons use the actual host build; they do not certify
an arbitrary IPP/custom-HAL build or substitute a portable independent oracle.
Historical CPU characterization is separate from GPU execution evidence, and an
older installed module does not gain new support because its source tree changed.

Unsupported codes/depths raise without CPU or `cv::cuda` fallback. This includes
reverse HSV, HLS, forward linear Luv, Bayer VNG/edge-aware, YUV encoding and other
size-changing codes. Odd-width 4:2:2 conversions and malformed 4:2:0 buffers are
rejected before launch. The two YUV gray extraction codes remain legal at odd widths.

The F32 arithmetic bound is relative to each output pixel's reference/input
scale; it is not a universal `1e-4` absolute bound at large magnitudes. A
consumer requiring that separate absolute cap must not infer it from a pass
under this scaled criterion. Integer conversions and float bit-copy families
retain their exact comparisons.

### `cv::triton::warpAffine`

Contract `warpaffine/0.4.3`: `CV_8U`, `CV_16U` and `CV_32F`; 1/3/4 channels; `INTER_NEAREST`, `INTER_LINEAR`
and `INTER_CUBIC`; `INTER_AREA` aliases `INTER_LINEAR`. The five standard border
modes and `BORDER_TRANSPARENT` for NEAREST/LINEAR/AREA are supported, with
`WARP_INVERSE_MAP` both ways.

| dimension | supported |
|---|---|
| depth | `CV_8U`, `CV_16U`, `CV_32F` |
| channels | 1, 3, 4 |
| interpolation | `INTER_NEAREST`, `INTER_LINEAR`, `INTER_CUBIC`; `INTER_AREA` aliases `INTER_LINEAR` |
| border modes | `CONSTANT`, `REPLICATE`, `REFLECT`, `REFLECT_101`, `WRAP`; `TRANSPARENT` for NEAREST/LINEAR/AREA |
| `borderValue` | a `Scalar` of up to four components, one per channel, kept in full precision |
| matrix | 2x3 `CV_32F` or `CV_64F`, forward map inverted internally unless `WARP_INVERSE_MAP` |

`BORDER_TRANSPARENT` reads the previous destination: NEAREST preserves it when
its rounded source point is outside; LINEAR/AREA substitute that destination
pixel for each outside tap with the original weight. Initialize a destination
of the requested size/type when these contents matter. `INTER_CUBIC` with
`BORDER_TRANSPARENT` is explicitly refused. C2 and `CV_16S` are also deliberately
refused: they need the CPU's other numerical path, rather than an input-check change.

`INTER_LANCZOS4` is **refused on purpose**: its 8x8 weight table and clamped
taps are a separate contract, and approximating them behind an unmeasured
tolerance is not acceptable. Also refused: `CV_64F`, channels outside 1/3/4, and
overlapping input/output.

The module identifies fused versus separate multiply/add through fixed probes
of the linked CPU implementation, separately for depth, channels and
`cv::useOptimized()` state. It does not use hardware FMA support as a proxy for
the compiled dispatch path. This bounded first-use calibration does not
compute the user's output on the CPU.

For `INTER_NEAREST` and `INTER_LINEAR` (including the `INTER_AREA` alias), what it computes is OpenCV's CPU
`cv::warpAffine` on its **vector path**, and this needs stating because that
build is not self-consistent: its SIMD body evaluates a column's coordinate as
`v_fma(M0, x, (float)(y*M1 + M2))` and its scalar tail as `x*M0 + y*M1 + M2`,
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
are part of the proposal and may change during review.
