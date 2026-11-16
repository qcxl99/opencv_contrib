Triton resize {#tutorial_triton_resize}
=============

@see @ref triton "Triton API reference".
@see @ref tutorial_triton_cuda_first_use "CUDA build and first use".
@see @ref tutorial_triton_hip_first_use "HIP / DTK and ROCm build and first use".

Supported inputs
----------------

`cv::triton::resize` supports `CV_8U` and `CV_32F`, with 1, 3 or 4 channels.
CUDA callers pass `GpuMat`; HIP callers pass caller-owned `DeviceImage` views.
Interpolation modes are `INTER_NEAREST`, `INTER_LINEAR`, `INTER_CUBIC` and
`INTER_AREA`. AREA supports integer and fractional ratios and upscaled axes.
Same-size requests copy the image without interpolation. Source and destination
may be pitched ROIs with contiguous pixels and channels; their pixel storage must
not overlap. Separate ROIs may share a parent allocation.
Correctly sized GpuMat output is reused; DeviceImage output must be preallocated.

An explicit positive `dsize` takes precedence over `fx` and `fy`. If `dsize` is
empty, both scales must be finite and positive and the rounded dimensions must
fit positive int32 values. The original scales are retained for coordinate
mapping. Sizing and interpolation are checked before output allocation.
`INTER_NEAREST_EXACT`, `INTER_LINEAR_EXACT`, warp flags, other depths and other
channel counts are unsupported. There is no CPU or `cv::cuda` fallback.

Numerical behavior
------------------

The reference is the generic OpenCV 5.0.0 CPU resize algorithm with 128-bit SIMD,
without FMA, IPP or HAL substitutions. `CV_8U` output is bit-exact against this
reference. `CV_32F` comparisons allow float32 rounding error according to the
operation-specific bounds in the accuracy tests. Tests use a portable CPU
implementation in the test sources so that host-library dispatch choices do
not change their expected output.

`INTER_LINEAR` at exact 2× downsampling uses the CPU AREA route. Other CPU
configurations and `cv::cuda::resize` may differ in coordinate mapping,
fixed-point rounding or route selection. Native CUDA timings therefore do not
establish numerical equivalence.

Calling the function
--------------------

The following CUDA example resizes a host image to half its dimensions:

```cpp
#include <opencv2/triton.hpp>

cv::cuda::GpuMat src(host), dst;  // host is a cv::Mat
cv::cuda::Stream stream;
cv::triton::resize(src, dst, cv::Size(host.cols / 2, host.rows / 2),
                   0, 0, cv::INTER_LINEAR, stream);
stream.waitForCompletion();
```

The function submits work asynchronously. Retain input and output allocations
until the stream completes. HIP applications use the `DeviceImage` overloads of
`cv::triton::resize` from `opencv2/triton/device.hpp` with preallocated output
and a native stream handle. See `samples/hip/device.cpp`
(`example_triton_device resize`) for allocation and data transfers.
The overload with `fx`/`fy` supports scale-derived sizing; the six-argument
form requires explicit `dsize`.

Runtime and deployment
----------------------

See @ref triton_runtime_toolchain for compiler interpreters, launch ABI, cache
keys, persisted artifacts and resource relocation. For graph capture and
asynchronous resource ownership, see @ref triton_architecture and
cv::triton::releaseGraphResources().
