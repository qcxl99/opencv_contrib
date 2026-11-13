Triton resize {#tutorial_triton_resize}
=============

@note PROPOSED optional module. The module name, public API and build switches
are part of a proposal under review and may change. For a quick start see the
module's `README.md`; this page is the full usage guide.

@see @ref triton "Triton API reference".

What it computes
----------------

OpenCV CPU `cv::resize` as computed by a pinned reference configuration: the
generic `resize.cpp` path of OpenCV 5.0.0, 128-bit SIMD, no FMA, no IPP, no
HAL. `CV_8U` matches it bit-exactly; `CV_32F` within a bound derived from that
configuration's own float32 rounding (the kernels are bit-exact in practice).
"CPU `cv::resize`" without the configuration is not one function: an IPP-HAL
build of the same version differs on about a quarter of test cases, so the tests
compare against a portable C++ oracle compiled into the test binary, not against
the host build's `cv::resize`.

`cv::cuda::resize` computes a different result (coordinate mapping, exact-2x
`INTER_LINEAR` routed to the AREA path, AREA upscale, uint8 fixed-point
rounding). It is a performance baseline only (`perf/`), with different semantics.
This module is **not a drop-in replacement for `cv::cuda::resize`**.

Supported subset (this milestone)
---------------------------------

| dimension | supported |
|---|---|
| depth | `CV_8U`, `CV_32F` |
| channels | 1, 3, 4 |
| sizing | explicit `dsize`, or `fx`/`fy` when `dsize` is empty (the caller's values are used, as in `cv::resize`) |
| routes | same size (a device copy, not a kernel launch); `INTER_NEAREST`; `INTER_LINEAR` (exact 2x down takes the AREA fast path, as on the CPU); `INTER_CUBIC`; `INTER_AREA` at integer and fractional ratios, and with an upscaled axis |
| layout | ROI source/destination, row-contiguous pixels and channels; the output is reused when it already has the right size and type |
| platform | Linux; CUDA (driver API, `sm_90` tested) or HIP (`TRITON_BACKEND=hip`, gfx936 tested; device-buffer entries) |

Unsupported depths, channels and interpolation modes raise
`cv::Error::StsUnsupportedFormat`. All four declared interpolations execute the
CPU-aligned kernels; there is no implicit fallback to CPU or to `cv::cuda`.
Windows is not built in this milestone. HIP uses the device-buffer entries described below.

How a call runs
---------------

1. Parameters are resolved as `cv::resize` does and the request is validated
   (sizes, steps, int32 index ranges, overlap).
2. The compiled kernel for the request's compile key (kernel, signature dtypes,
   channel count and the other constexprs, launch options, target, Triton
   version, kernel source hash) is taken from an in-process cache, else from the
   on-disk cache (`$OPENCV_TRITON_CACHE_DIR`, default `~/.cache/opencv/triton`)
   after verifying its manifest and binary hash, else compiled by
   `tools/compile_helper.py` in a **separate, Torch-free process** and then
   verified the same way. Python never runs inside the calling process.
3. Coordinate and coefficient tables are built in C++ by the same code as the
   test oracle, uploaded on the caller's stream, and reused; a call on another
   stream waits for the upload through an event.
4. The kernel is launched with the parameter list recorded in the manifest. The
   two trailing scratch pointers Triton appends are passed as null only for a
   compiler whose entry ABI has been validated and a kernel that needs no
   scratch memory; otherwise the module refuses to launch.
5. Modules and tables are kept alive until the launches that use them complete;
   evicting them from the caches frees nothing early.

The first call with a new key costs a compile (about 0.1-0.6 s, plus interpreter
start); a known key with a new image size needs no Python at all. Loading a
module into a context waits for that context's in-flight work to drain (a CUDA
driver property).

Build
-----

```bash
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules \
      -DBUILD_opencv_triton=ON -DWITH_CUDA=ON \
      -DTRITON_PYTHON_EXECUTABLE=<python that can import triton> \
      -DTRITON_PYTHONPATH=<where triton is, if not on its default path> <opencv>
```

The module is off by default; an ordinary build needs no Python, Triton or
PyTorch, and an enabled build never needs PyTorch. Nothing is compiled at OpenCV
build time. `make install` places `kernels/resize_cpu.py` and
`tools/compile_helper.py` under `share/opencv5/triton`, found at run time
relative to the library (so the install tree can be moved);
`OPENCV_TRITON_RESOURCES`, `OPENCV_TRITON_PYTHON`, `OPENCV_TRITON_PYTHONPATH` and
`OPENCV_TRITON_CACHE_DIR` override the defaults. Without an interpreter, cached
keys still run; a cache miss raises a toolchain error.

Calling
-------

```cpp
#include <opencv2/triton.hpp>

cv::cuda::GpuMat src(host), dst(h / 2, w / 2, host.type());
cv::cuda::Stream stream;
cv::triton::resize(src, dst, dst.size(), 0, 0, cv::INTER_LINEAR, stream);
stream.waitForCompletion();
```

The call is asynchronous: work is submitted to the given stream and the inputs
and output must stay alive until it completes.

Build-time compilation
----------------------

Only Triton 3.6.0 (CUDA 12.8, sm_90) is verified. The compile key carries the
compiler version, so an artifact from another Triton is never reused; a build
with a different version is untested and should not be expected to work.

Nothing is compiled at OpenCV build time. On a cache miss the module runs
`tools/compile_helper.py` in a separate, Torch-free process and stores the
result in a verified on-disk cache; that process needs Python and Triton, the
call path does not.

An install-time cache prefill for deployments that cannot have Python at run
time is a planned item, not something this submission provides.
