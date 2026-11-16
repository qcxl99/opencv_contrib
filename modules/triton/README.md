# Triton-backed GPU operations

The optional Triton module provides `cv::triton::resize`, `cv::triton::cvtColor`
and `cv::triton::warpAffine`. Kernels are written in
[Triton](https://github.com/triton-lang/triton) and compiled on first use by a
separate Python process. The C++ execution path uses the compiled device code.
PyTorch is not required.

The module is disabled by default. Enable it with `BUILD_opencv_triton=ON`.
Calls use explicit `cv::triton` entry points and do not replace OpenCV functions.
Unsupported requests raise an exception without a CPU or `cv::cuda` fallback.
Operations are asynchronous on the supplied stream. Keep source and destination
allocations alive until the stream completes.

## Supported operations

| Operation | Depths | Channels | Modes |
|---|---|---|---|
| resize | `CV_8U`, `CV_32F` | 1, 3, 4 | NEAREST, LINEAR, CUBIC, AREA; same-size copy |
| cvtColor | `CV_8U`, selected `CV_16U` and `CV_32F` conversions | determined by conversion code | see the conversion table below |
| warpAffine | `CV_8U`, `CV_16U`, `CV_32F` | 1, 3, 4 | NEAREST, LINEAR, CUBIC; AREA aliases LINEAR |

Pitched images, source and destination ROIs, and output reuse are supported.
Source and destination must not overlap. Input validation checks sizes, strides,
indexing ranges, types, device ownership and stream compatibility.

Resize follows the generic OpenCV CPU algorithm described in the
[resize guide](doc/resize.markdown). Its reference uses OpenCV 5.0.0 with
128-bit SIMD and without FMA, IPP or HAL substitutions. `CV_8U` is exact against
that reference; float comparisons allow rounding error. Other CPU build
configurations and `cv::cuda::resize` can produce different results.

WarpAffine supports CONSTANT, REPLICATE, REFLECT, REFLECT_101 and WRAP borders,
and TRANSPARENT for NEAREST/LINEAR/AREA. CUBIC with TRANSPARENT and LANCZOS4 are
unsupported. NEAREST/LINEAR/AREA use the linked CPU implementation's vector-path
arithmetic, identified on first use; CUBIC uses the natural-width CPU reference.
This distinction can affect samples at rounding boundaries. For TRANSPARENT,
provide an initialized destination of the requested size and type. See the
[warpAffine guide](doc/warpaffine.markdown) for sampling and accuracy details.

### Colour conversions

There are 84 distinct numeric codes at `CV_8U`, 20 at `CV_16U` and 20 at
`CV_32F`. Enum aliases share a code. Support at one depth does not imply support
at another depth.

| Conversion family | Channels | Depths |
|---|---|---|
| BGR/BGRA/RGB/RGBA channel permutations (six codes) | 3/4 → 3/4 | 8U, 32F |
| BGR/RGB/BGRA/RGBA to gray | 3/4 → 1 | 8U, 16U, 32F |
| Gray to BGR/BGRA | 1 → 3/4 | 8U, 16U, 32F |
| YCrCb and YUV, BGR/RGB in both directions | 3 → 3 | 8U, 16U, 32F |
| Forward HSV and HSV_FULL, BGR/RGB | 3 → 3 | 8U |
| Packed BGR565/BGR555 encode and decode | 1/3/4 ↔ 2 | 8U |
| Bilinear Bayer: BG/GB/RG/GR to BGR, BG to BGRA/GRAY | 1 → 1/3/4 | 8U, 16U |
| XYZ (four), Lab (eight), Luv (six, excluding forward linear Luv) | 3 → 3 | 8U |
| UYVY/YUY2/YVYU to BGR/RGB | 2 → 3 | 8U, even width |
| YUV2GRAY_UYVY and YUV2GRAY_YUY2 | 2 → 1 | 8U, odd width allowed |
| NV12/NV21 to BGR/RGB/BGRA/RGBA | 1 → 3/4 | 8U, packed `(3H/2) × W` input |

The conversion code determines channel counts; there is no `dstCn` override.
Packed 565/555 uses `CV_8UC2`. NV12/NV21 requires even width and a source row
count divisible by three. Other conversions preserve image size. Reverse HSV,
HLS, forward linear Luv, Bayer VNG/edge-aware and video encoding are unsupported.
Integer conversions and float channel copies are exact against the linked CPU
implementation. Float arithmetic uses a per-pixel error bound, specified in the
[cvtColor guide](doc/cvtcolor.markdown).

## Build

Enable this default-off module with `BUILD_opencv_triton=ON` in a matching
OpenCV/contrib out-of-source build. PyTorch is not required. Choose the backend
tutorial for build, install, device selection and first-use commands:

- [CUDA](tutorials/cuda-first-use.markdown): GpuMat calls and Python bindings;
  H20/CUDA is a configuration example.
- [HIP](tutorials/hip-first-use.markdown): C++ DeviceImage calls; DTK/gfx936 and
  ROCm/gfx1100 configuration differences are listed separately.

The runtime detects the actual target; configuration examples do not certify
other machines or revisions. Compiler/ABI, cache and relocation details are
maintained in the [runtime toolchain section](doc/maintenance.markdown#triton_runtime_toolchain).

## Use

CUDA builds provide GpuMat overloads and Python bindings:

```cpp
#include <opencv2/triton.hpp>

cv::cuda::GpuMat src(host), dst;
cv::cuda::Stream stream;
cv::triton::resize(src, dst, cv::Size(host.cols / 2, host.rows / 2),
                   0, 0, cv::INTER_LINEAR, stream);
stream.waitForCompletion();
cv::Mat result;
dst.download(result);
```

```python
import cv2 as cv

src = cv.cuda_GpuMat(image)
small = cv.triton.resize(src, (640, 360), interpolation=cv.INTER_LINEAR)
gray = cv.triton.cvtColor(small, cv.COLOR_BGR2GRAY)
result = gray.download()
```

HIP applications use the caller-owned `DeviceImage` overloads in
`opencv2/triton/device.hpp`: `resize`, `cvtColor` and
`warpAffine`. They do not allocate output storage or synchronize.
The scale-aware resize overload accepts empty `dsize` with `fx`/`fy`; the caller
must preallocate the resolved destination size. GpuMat operations are unavailable
on HIP and raise `StsNotImplemented`, including Python calls with omitted or
`None` streams. Backend queries remain available in Python.

HIP allocation device/range checks are common to both configurations. DTK can
query native context generation and stream context. On ROCm without those
extensions, the runtime supports retained primary contexts and checks the
stream's device. Arbitrary foreign-context allocation/stream ownership is not
proven by those fallback queries. See the corresponding platform guide.

## Samples and tests

Enable `BUILD_EXAMPLES`, `BUILD_TESTS` and `BUILD_PERF_TESTS` as needed.
CUDA samples are `example_triton_resize`, `example_triton_cvtcolor` and
`example_triton_warpaffine`. HIP uses
`example_triton_device [resize|warp|cvtcolor]`. Each sample demonstrates explicit
stream use and result download.

`opencv_test_triton` runs common accuracy tests through the selected backend.
CUDA builds include CUDA lifecycle tests; HIP additionally provides
`opencv_test_triton_hip` for lifecycle and validation-error tests.
`opencv_perf_triton` measures warmed public calls including stream completion.
Optional native CUDA comparisons require `cudawarping` and `cudaimgproc`; these
are not production dependencies. Such timings use algorithms with potentially
different numerical behavior and do not replace accuracy tests.

The optional [HIP graph benchmark](perf/hip/GRAPH.md) measures warmed replay
separately from ordinary calls.

See [architecture and maintenance](doc/maintenance.markdown),
[validation methods](doc/validation.markdown), and the
[API header](include/opencv2/triton.hpp).
