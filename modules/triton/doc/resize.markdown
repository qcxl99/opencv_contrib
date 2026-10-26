Triton resize {#tutorial_triton_resize}
=============

@note PROPOSED optional module. The module name, public API and build switches
are a design proposal and have not been confirmed with OpenCV maintainers.

Supported subset
----------------

| dimension | supported |
|---|---|
| depth | `CV_8U` |
| channels | 1, 3 |
| sizing | explicit `dsize` (the `fx`/`fy` path is rejected) |
| interpolation | `INTER_NEAREST`, `INTER_LINEAR` |
| layout | non-ROI source, row-contiguous pixels and channels |

Anything else raises `cv::Error::StsUnsupportedFormat`. There is **no implicit
fallback** to CPU or to `cv::cuda`: a call that returns has executed the Triton
kernel. `cv::cuda::resize` is untouched and remains available as an independent
performance reference.

Semantics match `cv::cuda::resize` **bit-exactly** for that subset. They do not
match OpenCV CPU `resize`: for `INTER_LINEAR` the CUDA rule computes
`src_x = dst_x * fx` with no half-pixel shift, so CPU and CUDA disagree by
design. Tests therefore compare against the CUDA path.

Build
-----

```bash
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules \
      -DWITH_TRITON=ON -DTRITON_BACKEND_CUDA=ON \
      -DPYTHON3_EXECUTABLE=<python with triton> <opencv>
```

With `WITH_TRITON=OFF` (the default) an ordinary build needs no Python, Triton
or PyTorch. With it ON, kernels are compiled **at build time** into cubins plus
`kernels_manifest.json` under `<build>/modules/triton/generated/`; the manifest
records the kernel source hash, Triton version, target architecture and
parameter ABI so an artifact is traceable and rebuildable.

Image height, width and row pitch stay runtime arguments, so no artifact is
generated per image size. Variants exist only for the dimensions Triton
actually specializes on: interpolation mode, channel count, and
divisibility-by-16 of both the pointer and the byte stride — 64 in total.

Calling
-------

```cpp
#include <opencv2/triton.hpp>

cv::cuda::GpuMat src(host), dst(h / 2, w / 2, host.type());
cv::cuda::Stream stream;
cv::triton::resize(src, dst, dst.size(), 0, 0, cv::INTER_LINEAR, stream);
stream.waitForCompletion();
```

The call is asynchronous: work is submitted to the given stream and the inputs,
output and any temporaries must stay alive until it completes. The normal path
never calls `cudaDeviceSynchronize` for convenience.

Limitations
-----------

- H20 (`sm_90`) only. No Hygon BW1000 backend yet; CUDA artifacts and loading
  are not portable to it, and a second backend must be added under
  `src/backend/hygon/` after real-hardware probing.
- A build-time Python interpreter with Triton is required when the module is
  enabled; it is not needed at runtime.
- `fx`/`fy`, ROI sources, `CV_32F`, 4 channels and other interpolations are
  out of scope for this round and are rejected rather than emulated.
