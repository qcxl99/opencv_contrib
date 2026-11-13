cvtColor: supported conversions and execution {#tutorial_triton_cvtcolor}
============================================

The optional Triton module provides explicit `cv::triton::cvtColor` calls.
The [module README](../README.md) lists the conversion families. The native
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

Integer conversions and float channel copies compare exactly against the
linked OpenCV CPU implementation. Finite float gray/YUV/YCrCb arithmetic uses
`8 * 2^-24 * max(1, abs(ref), max(abs(input channels at this pixel)))` as the
absolute error bound for each output element. Another pixel's magnitude does
not increase the allowance. IPP/custom HAL builds require their own evidence.

CUDA uses GpuMat and the public Python bindings. HIP uses the C++ DeviceImage
entries in `opencv2/triton/device.hpp`; `example_triton_device cvtcolor` shows
allocation, upload, explicit stream submission, synchronization and comparison.
All three Python GpuMat adapters (resize, cvtColor and warpAffine) return
catchable StsNotImplemented on HIP, including default/None streams.

Calls are asynchronous. Callers retain their images and order producers and
consumers. Compiled modules and HSV/Lab/Luv tables use the runtime's stream and
in-flight retention machinery. Warm graph capture is tested; after destroying
graphs, call the public releaseGraphResources as documented in the API header.
Bayer passes image height explicitly in its JIT ABI on both backends.

Accuracy tests cover every admitted code/depth pair, boundary colours, short
axes and SIMD/block-boundary widths, pitched ROIs, output guards, refusals and
lookup-table stream/capture lifetime. CUDA Python numerical tests require an
actual CUDA build; HIP binding tests check the documented safe refusals.
OpenCV perf tests provide representative family/depth rows with warmup;
compilation and allocation are excluded. Public-call wall time includes host
submission and synchronization; it is not isolated kernel or graph time.
