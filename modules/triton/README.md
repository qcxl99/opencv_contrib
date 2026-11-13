# Triton-backed GPU operations

An optional opencv_contrib module whose image operators are written in
[Triton](https://github.com/triton-lang/triton) and compiled on first use at run
time. It provides `cv::triton::resize`, `cv::triton::cvtColor` and
`cv::triton::warpAffine`.

The module is **off by default** and does not modify any existing OpenCV
function. `cv::cuda::resize`, `cv::cuda::cvtColor` and `cv::cuda::warpAffine` are
untouched; a caller opts in by name by calling the `cv::triton::` entry.

## What it supports

CUDA GpuMat entries: Linux, CUDA (driver API), `sm_90` tested; ROI source and destination;
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

The module serves **84 distinct numeric conversion codes** (enum aliases
share a code). Supported conversions are:

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

The module identifies fused versus separate multiply/add through fixed probes
of the linked CPU implementation, separately for depth, channels and
`cv::useOptimized()` state. It does not use hardware FMA support as a proxy for
the compiled dispatch path. This bounded first-use calibration does not
compute the user's output on the CPU.

For `INTER_NEAREST` and `INTER_LINEAR` (including the `INTER_AREA` alias), what it computes is OpenCV's CPU
`cv::warpAffine` on its **vector path**, and this needs stating because that
build can depend on destination width: its SIMD body evaluates a column's coordinate as
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

Use the normal OpenCV out-of-source build with this contrib checkout:

```bash
cmake -S <opencv> -B <build> \
      -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules \
      -DCMAKE_INSTALL_PREFIX=<private-install> \
      -DBUILD_opencv_triton=ON -DWITH_CUDA=ON -DTRITON_BACKEND=cuda \
      -DTRITON_PYTHON_EXECUTABLE=<python-that-imports-triton>
cmake --build <build> --parallel 8
cmake --install <build>
```

Replace the path placeholders. For a complete Ubuntu/H20 walkthrough, including
Python development files/bindings, explicit `sm_90`, tests, samples, actual import
identity and cold/hot cache checks, see [H20 first use](doc/h20-first-use.markdown).
Enable `BUILD_TESTS`, `BUILD_PERF_TESTS` and `BUILD_EXAMPLES` when those targets
are required; `INSTALL_TESTS=ON` installs test programs. Python bindings use
OpenCV's standard `python3` / `python_bindings_generator` build configuration;
`TRITON_PYTHON_EXECUTABLE` independently selects the cold-path compiler process.
`TRITON_PYTHONPATH` is optional when that interpreter imports the intended Triton
without an extra search path.

With this module off, it adds no Python, Triton or PyTorch requirement; other
OpenCV modules retain their own dependencies. An enabled module does not require
PyTorch. **Verified compiler ABI: Triton 3.6.0, with CUDA 12.8 / sm_90 as the
recorded CUDA target.** Compiler version participates in the compile key; the
launcher rejects unvalidated ABIs rather than reusing incompatible artifacts.

A cold artifact cache needs Python and Triton in a separate Torch-free helper.
The hot path is C++ and does not enter an interpreter. A writable cache is needed
at runtime; `OPENCV_TRITON_CACHE_DIR` selects an isolated location. See the
[resize guide](doc/resize.markdown) for resources, compiler overrides and errors.

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
results, with your own device buffers and a `hipStream_t`. Standard
`opencv_test_triton` and `opencv_perf_triton` use the common cases/references via
the HIP validation adapter. `opencv_test_triton_hip` adds HIP lifecycle/refusal
groups. CUDA-specific legacy tests and GpuMat samples remain CUDA-only; the HIP
sample is selected from `samples/hip/`.

### HIP examples and Python boundary

With `BUILD_EXAMPLES=ON`, `example_triton_device resize`,
`example_triton_device warp` and `example_triton_device cvtcolor` demonstrate
caller-owned HIP buffers, an explicit stream, synchronization and a CPU
comparison. These examples require no Torch dependency.

The Python module can be built and imported on HIP, and its backend queries
work. Its resize, cvtColor and warpAffine GpuMat adapters share a nullable-stream
bridge: omitting the stream or passing None raises a catchable
`StsNotImplemented`. HIP numerical calls currently use the C++ device-buffer
entries; numerical Python GpuMat adapters remain unsupported.

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

Three CUDA buildable examples, `samples/resize.cpp`, `samples/cvtcolor.cpp` and
`samples/warpaffine.cpp`; all are
built by `-DBUILD_EXAMPLES=ON` and need a Triton backend at run time:

```bash
./bin/example_triton_cvtcolor input.png gray.png     # BGR -> GRAY
./bin/example_triton_cvtcolor                        # synthetic input
./bin/example_triton_resize input.png out.png        # half size, INTER_LINEAR
./bin/example_triton_warpaffine                      # synthetic input
```

## Further reading

- [Module maintenance](doc/maintenance.markdown) — source ownership, local generation,
  shared validation and evidence boundaries
- [BW1000 first use](doc/bw1000-first-use.markdown) — HIP build, tests, perf and deployment

- [H20 first use](doc/h20-first-use.markdown) — standard source build/install,
  C++ and Python use, native tests and deployment checks on Ubuntu/H20
- [resize guide](doc/resize.markdown) — resize in depth: dependencies, the JIT
  and its cache, the numerical contract, streams, limitations and troubleshooting.
  Everything except the resize-specific sections applies to `cvtColor` and
  `warpAffine` as well
- [cvtColor guide](doc/cvtcolor.markdown) — depth-specific conversions, video layouts,
  numerical bounds and HIP/Python boundaries
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
platform (Hygon BW1000, gfx936, DTK 26.04, flagtree 3.6.0); its C++ device-buffer routes include four resize modes, the documented
warpAffine modes/borders, and all 124 cvtColor code/depth pairs. Validation
support selects real CUDA or HIP public APIs and shares cases and numerical
criteria; results remain specific to each backend.

## Backend integration boundary

The current CUDA API and Python calls are unchanged. Private output resolution,
request validation and error mapping are shared by adapters; CUDA owns GpuMat
allocation and verifies buffer/stream context ownership. Kernels, runtime JIT,
cache identities and numerical contracts are unchanged by this refactoring.

Tests and Triton candidate perf cases use private backend support described in
[validation/README.md](validation/README.md). Existing CUDA-specific compatibility
and lifecycle tests remain. This supplies a reuse boundary for other platforms;
it does not claim a unified public DeviceMat/Stream API or transfer H20 acceptance to HIP.
Future UMat/non-CPU HAL integration would provide another thin adapter after
OpenCV finalizes its extension interfaces. This module does not modify or
intercept standard `cv::*` calls; unsupported inputs raise rather than fall back.


### HIP validation and scale semantics

The HIP backend builds standard `opencv_test_triton` common accuracy tests with
`BUILD_TESTS=ON` and standard `opencv_perf_triton` candidate perf rows with
`BUILD_PERF_TESTS=ON`. CUDA-specific tests/baselines are not compiled into those
HIP targets. `opencv_test_triton_hip` supplements them with five registered
backend lifecycle/refusal groups; its XML records inner-check counts separately.
All three HIP examples use `example_triton_device [resize|warp|cvtcolor]`.

The existing six-argument `resizeDevice` requires explicit dsize. Its scale-aware
overload additionally takes fx/fy before interpolation; when dsize is empty the
original scales reach the calculation unchanged. An explicit dsize takes
precedence. The caller preallocates dst to the resolved size; the API does not
allocate, copy to contiguous storage, change streams or synchronize implicitly.
This adds parameter parity, not a public memory-management layer.

The HIP SDK verifies allocation device/range plus active context generation and
stream context. DTK does not expose allocation context through
HIP_POINTER_ATTRIBUTE_CONTEXT, so arbitrary foreign driver-context allocation
ownership is not an advertised verified capability.
