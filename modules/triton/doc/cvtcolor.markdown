Triton cvtColor {#tutorial_triton_cvtcolor}
===============

The optional Triton module provides explicit `cv::triton::cvtColor` calls.
CUDA callers pass `GpuMat`; HIP callers pass caller-owned `DeviceImage` views.
The table below lists every supported conversion. The native
code table contains 84 distinct numeric codes at CV_8U, 20 at CV_16U and 20 at
CV_32F. Enum aliases share a row. A supported code at one depth does not imply
support at another depth; unsupported combinations raise without CPU fallback.

The code fixes the source and destination channel counts. There is no dstCn
override. Channel permutations, colour/gray, YUV/YCrCb, forward HSV, packed
565/555, bilinear Bayer, XYZ/Lab/Luv and video decode are covered as listed.
Reverse HSV/HLS, forward linear Luv, Bayer VNG/EA and video encoding are refused.

Packed 565/555 use CV_8UC2. Interleaved UYVY/YUY2/YVYU conversion requires even
width; the two Y-channel extraction codes permit odd width. NV12/NV21 take a
single row-pitched CV_8UC1 image with an even width and a height divisible by 3:
its first H rows are luma, followed by H/2 rows of interleaved chroma; output is
H by W. ROI pitch is respected for both planes. Invalid geometry and offsets
outside the runtime integer indexing range are refused.

Calling the function
--------------------

The following CUDA example converts a host BGR image to device-resident gray:

```cpp
#include <opencv2/triton.hpp>

cv::cuda::GpuMat src(host), gray;  // host is a CV_8UC3 cv::Mat
cv::cuda::Stream stream;
cv::triton::cvtColor(src, gray, cv::COLOR_BGR2GRAY, stream);
stream.waitForCompletion();
```

The destination is allocated or reused. Calls are asynchronous; retain both
allocations until the stream completes. HIP uses the `DeviceImage` overload
from `opencv2/triton/device.hpp`, with preallocated output and a native stream
handle. The complete allocation/upload/download example is
`samples/hip/device.cpp` (`example_triton_device cvtcolor`).

Numerical behavior
------------------

Integer conversions and float channel copies compare exactly against the
linked OpenCV CPU implementation. Finite float gray/YUV/YCrCb arithmetic uses
`8 * 2^-24 * max(1, abs(ref), input_scale)` as the absolute error bound for
each output element. Here, `input_scale` is the largest absolute input-channel
value at the corresponding pixel. Another pixel's magnitude does not increase
the allowance. IPP or custom HAL substitutions can change CPU results.

When the linked CPU implementation rounds multiplication and addition separately
and processes groups of four values, the runtime selects matching GPU arithmetic.
The accuracy suite then requires absolute error at most `1e-4` as well as the
scaled bound. Identification executes
small directed fixtures against the actual linked `cv::cvtColor`; it never sends
user images to CPU or substitutes probe output for user output. The process-local
cache distinguishes optimization, IPP/exactness and floating rounding controls;
unstable probes or non-default rounding do not identify a profile. Explicit
rounded GPU operations and per-row vector-body/scalar-tail grouping preserve
this expression instead of relying on compiler FMA contraction settings.

Directed accuracy fixtures include unit/255 scale and adjacent signed values
through `2*65535`, short rows, vector-boundary widths and pitched ROIs.
They do not establish a universal bound for overflow, subnormals, other HALs
or changed arithmetic environments. An unidentified profile uses the fused path
and the scaled bound without the additional absolute cap. Copy paths remain exact.


CUDA provides GpuMat Python bindings. The HIP `DeviceImage` overload is C++ only.
All three Python GpuMat adapters (resize, cvtColor and warpAffine) return
catchable `StsNotImplemented` on HIP, including default/`None` streams.

Calls are asynchronous. Callers retain their images and order producers and
consumers. The runtime keeps compiled modules and HSV/Lab/Luv tables alive
until submitted work finishes.

Warm operations support graph capture. After all graphs using these resources
are destroyed or will never be replayed again, call
cv::triton::releaseGraphResources() as documented in the API reference.

Accuracy tests cover every admitted code/depth pair, boundary colours, short
axes and SIMD/block-boundary widths, pitched ROIs, output guards, refusals and
lookup-table stream/capture lifetime. CUDA Python numerical tests require an
actual CUDA build; HIP binding tests check the documented safe refusals.
OpenCV performance tests provide representative family/depth cases with warmup;
compilation and allocation are excluded. Public-call wall time includes host
submission and synchronization; it is not isolated kernel or graph time.


Supported conversion codes
--------------------------

Each row names one numeric code. OpenCV enum aliases with the same value are
also accepted; aliases do not add conversions. Depth support is specific to each row.

| Conversion code | Source → output channels | Depths |
|---|---|---|
| `COLOR_BGR2BGRA` | 3 → 4 | 8U, 32F |
| `COLOR_BGRA2BGR` | 4 → 3 | 8U, 32F |
| `COLOR_BGR2RGBA` | 3 → 4 | 8U, 32F |
| `COLOR_RGBA2BGR` | 4 → 3 | 8U, 32F |
| `COLOR_BGR2RGB` | 3 → 3 | 8U, 32F |
| `COLOR_BGRA2RGBA` | 4 → 4 | 8U, 32F |
| `COLOR_GRAY2BGR` | 1 → 3 | 8U, 16U, 32F |
| `COLOR_GRAY2RGBA` | 1 → 4 | 8U, 16U, 32F |
| `COLOR_BGR2GRAY` | 3 → 1 | 8U, 16U, 32F |
| `COLOR_RGB2GRAY` | 3 → 1 | 8U, 16U, 32F |
| `COLOR_BGRA2GRAY` | 4 → 1 | 8U, 16U, 32F |
| `COLOR_RGBA2GRAY` | 4 → 1 | 8U, 16U, 32F |
| `COLOR_YUV2GRAY_UYVY` | 2 → 1 | 8U |
| `COLOR_YUV2GRAY_YUY2` | 2 → 1 | 8U |
| `COLOR_YUV2BGR_NV12` | 1 → 3 | 8U |
| `COLOR_YUV2RGB_NV12` | 1 → 3 | 8U |
| `COLOR_YUV2BGRA_NV12` | 1 → 4 | 8U |
| `COLOR_YUV2RGBA_NV12` | 1 → 4 | 8U |
| `COLOR_YUV2BGR_NV21` | 1 → 3 | 8U |
| `COLOR_YUV2RGB_NV21` | 1 → 3 | 8U |
| `COLOR_YUV2BGRA_NV21` | 1 → 4 | 8U |
| `COLOR_YUV2RGBA_NV21` | 1 → 4 | 8U |
| `COLOR_BGR2YCrCb` | 3 → 3 | 8U, 16U, 32F |
| `COLOR_RGB2YCrCb` | 3 → 3 | 8U, 16U, 32F |
| `COLOR_YCrCb2BGR` | 3 → 3 | 8U, 16U, 32F |
| `COLOR_YCrCb2RGB` | 3 → 3 | 8U, 16U, 32F |
| `COLOR_BGR2HSV` | 3 → 3 | 8U |
| `COLOR_RGB2HSV` | 3 → 3 | 8U |
| `COLOR_BGR2HSV_FULL` | 3 → 3 | 8U |
| `COLOR_RGB2HSV_FULL` | 3 → 3 | 8U |
| `COLOR_YUV2RGB_UYVY` | 2 → 3 | 8U |
| `COLOR_YUV2BGR_UYVY` | 2 → 3 | 8U |
| `COLOR_YUV2RGB_YUY2` | 2 → 3 | 8U |
| `COLOR_YUV2BGR_YUY2` | 2 → 3 | 8U |
| `COLOR_YUV2RGB_YVYU` | 2 → 3 | 8U |
| `COLOR_YUV2BGR_YVYU` | 2 → 3 | 8U |
| `COLOR_BGR2YUV` | 3 → 3 | 8U, 16U, 32F |
| `COLOR_RGB2YUV` | 3 → 3 | 8U, 16U, 32F |
| `COLOR_YUV2BGR` | 3 → 3 | 8U, 16U, 32F |
| `COLOR_YUV2RGB` | 3 → 3 | 8U, 16U, 32F |
| `COLOR_BGR2BGR565` | 3 → 2 | 8U |
| `COLOR_RGB2BGR565` | 3 → 2 | 8U |
| `COLOR_BGRA2BGR565` | 4 → 2 | 8U |
| `COLOR_RGBA2BGR565` | 4 → 2 | 8U |
| `COLOR_GRAY2BGR565` | 1 → 2 | 8U |
| `COLOR_BGR5652BGR` | 2 → 3 | 8U |
| `COLOR_BGR5652RGB` | 2 → 3 | 8U |
| `COLOR_BGR5652BGRA` | 2 → 4 | 8U |
| `COLOR_BGR5652RGBA` | 2 → 4 | 8U |
| `COLOR_BGR5652GRAY` | 2 → 1 | 8U |
| `COLOR_BGR2BGR555` | 3 → 2 | 8U |
| `COLOR_RGB2BGR555` | 3 → 2 | 8U |
| `COLOR_BGRA2BGR555` | 4 → 2 | 8U |
| `COLOR_RGBA2BGR555` | 4 → 2 | 8U |
| `COLOR_GRAY2BGR555` | 1 → 2 | 8U |
| `COLOR_BGR5552BGR` | 2 → 3 | 8U |
| `COLOR_BGR5552RGB` | 2 → 3 | 8U |
| `COLOR_BGR5552BGRA` | 2 → 4 | 8U |
| `COLOR_BGR5552RGBA` | 2 → 4 | 8U |
| `COLOR_BGR5552GRAY` | 2 → 1 | 8U |
| `COLOR_BayerBG2BGR` | 1 → 3 | 8U, 16U |
| `COLOR_BayerGB2BGR` | 1 → 3 | 8U, 16U |
| `COLOR_BayerRG2BGR` | 1 → 3 | 8U, 16U |
| `COLOR_BayerGR2BGR` | 1 → 3 | 8U, 16U |
| `COLOR_BayerBG2GRAY` | 1 → 1 | 8U, 16U |
| `COLOR_BayerBG2BGRA` | 1 → 4 | 8U, 16U |
| `COLOR_BGR2XYZ` | 3 → 3 | 8U |
| `COLOR_RGB2XYZ` | 3 → 3 | 8U |
| `COLOR_XYZ2BGR` | 3 → 3 | 8U |
| `COLOR_XYZ2RGB` | 3 → 3 | 8U |
| `COLOR_BGR2Lab` | 3 → 3 | 8U |
| `COLOR_RGB2Lab` | 3 → 3 | 8U |
| `COLOR_Lab2BGR` | 3 → 3 | 8U |
| `COLOR_Lab2RGB` | 3 → 3 | 8U |
| `COLOR_BGR2Luv` | 3 → 3 | 8U |
| `COLOR_RGB2Luv` | 3 → 3 | 8U |
| `COLOR_Luv2BGR` | 3 → 3 | 8U |
| `COLOR_Luv2RGB` | 3 → 3 | 8U |
| `COLOR_LBGR2Lab` | 3 → 3 | 8U |
| `COLOR_LRGB2Lab` | 3 → 3 | 8U |
| `COLOR_Lab2LBGR` | 3 → 3 | 8U |
| `COLOR_Lab2LRGB` | 3 → 3 | 8U |
| `COLOR_Luv2LBGR` | 3 → 3 | 8U |
| `COLOR_Luv2LRGB` | 3 → 3 | 8U |
