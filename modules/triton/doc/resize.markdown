Triton resize {#tutorial_triton_resize}
=============

@see @ref triton "Triton API reference".
@see @ref tutorial_triton_h20_first_use "CUDA build and first use".
@see @ref tutorial_triton_bw1000_first_use "HIP build and first use".

Supported inputs
----------------

`cv::triton::resize` supports `CV_8U` and `CV_32F`, with 1, 3 or 4 channels.
Interpolation modes are `INTER_NEAREST`, `INTER_LINEAR`, `INTER_CUBIC` and
`INTER_AREA`. AREA supports integer and fractional ratios and upscaled axes.
Same-size requests copy the image without interpolation. Source and destination
may be pitched ROIs with contiguous pixels and channels; their pixel storage must
not overlap. Separate ROIs may share a parent allocation.
Correctly sized output storage is reused.

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

```cpp
#include <opencv2/triton.hpp>

cv::cuda::GpuMat src(host), dst;
cv::cuda::Stream stream;
cv::triton::resize(src, dst, cv::Size(host.cols / 2, host.rows / 2),
                   0, 0, cv::INTER_LINEAR, stream);
stream.waitForCompletion();
```

The function submits work asynchronously. Retain input and output allocations
until the stream completes. HIP applications use `resizeDevice` from
`opencv2/triton/device.hpp` with preallocated output and a native stream handle.
The overload with `fx`/`fy` supports scale-derived sizing; the six-argument
form requires explicit `dsize`.

Runtime compilation and cache
-----------------------------

The module is disabled by default and requires no PyTorch dependency when
enabled. Nothing is compiled by Triton during the OpenCV build. On first use,
`tools/compile_helper.py` runs in a separate Python process using the configured
Triton 3.6.0 compiler. The C++ execution path does not embed an interpreter.
CUDA 12.8/sm_90 and the HIP flagtree 3.6.0 toolchain are described in the
platform guides linked above.

The compile key includes source identity, kernel signature, compile-time
parameters, launch options, target and compiler version. The runtime checks
manifest and binary hashes before loading disk artifacts. An unsupported
compiler launch ABI is rejected. Size-dependent coordinate and coefficient
tables are built in C++, uploaded on the caller's stream and cached. Cross-stream
use waits on the upload event. In-flight work retains modules and tables even
after cache eviction.

The writable cache defaults to `~/.cache/opencv/triton`. The following runtime
settings override configured defaults:

| Setting | Purpose |
|---|---|
| `OPENCV_TRITON_CACHE_DIR` | Artifact cache directory |
| `OPENCV_TRITON_RESOURCES` | Kernel and helper resource directory |
| `OPENCV_TRITON_PYTHON` | Compiler interpreter |
| `OPENCV_TRITON_PYTHONPATH` | Compiler Python import path |

Installation places resources under `share/opencv5/triton`, resolved relative
to the module library. A disk-cache hit runs without Python; a cache miss with
no usable compiler raises a toolchain error. An empty module artifact cache
does not imply that the compiler's own cache is empty.

Warm operations can be captured into device graphs. After graphs using module
resources are destroyed or can no longer be replayed, call
cv::triton::releaseGraphResources(). Do not release those resources while a
graph can still use them.
