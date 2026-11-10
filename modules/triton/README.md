# Triton-backed GPU operations

An optional opencv_contrib module whose image operators are written in
[Triton](https://github.com/triton-lang/triton) and compiled on first use at run
time. The first release provides one operator, `cv::triton::resize`.

The module is **off by default** and does not modify any existing OpenCV
function. `cv::cuda::resize` is untouched, and a caller opts in by name by calling
`cv::triton::resize`.

## What it supports

| | |
|---|---|
| platform | Linux, CUDA (driver API), `sm_90` tested |
| depth | `CV_8U`, `CV_32F` |
| channels | 1, 3, 4 |
| sizing | explicit `dsize`, or `fx`/`fy` when `dsize` is empty |
| routes | same size (a device copy), `INTER_NEAREST`, `INTER_LINEAR`, `INTER_AREA` at integer ratios or with an upscaled axis |
| layout | ROI source and destination; output reused when it already has the right size and type |

Unsupported resize routes, depths and channel counts raise
`cv::Error::StsUnsupportedFormat`; invalid layouts, in-place operations and
overlapping src/dst are rejected as invalid arguments. A same-size copy does
not perform interpolation, including when `INTER_CUBIC` is requested.
**There is no implicit fallback** to CPU or to `cv::cuda`: a call that returns
has submitted the computation or a same-size copy.

What it computes is OpenCV **CPU** `cv::resize` of a pinned reference
configuration, so uint8 output matches it bit-exactly. It is not a drop-in
replacement for `cv::cuda::resize`, which computes a different result.

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
```

The call is asynchronous on the stream you pass; inputs and output must stay
alive until the work completes.

## Further reading

- [Full usage guide](doc/resize.markdown) — dependencies, the JIT and its cache,
  the numerical contract, streams, limitations and troubleshooting
- [Buildable sample](samples/resize.cpp)
- [API reference](include/opencv2/triton.hpp)
- Tests: `test/test_resize.cpp` (accuracy), `test/test_runtime.cpp` (lifetime and
  caching), `misc/python/test/test_triton.py` (binding)

## Status

Proposed for opencv_contrib; not yet merged. The public API and build switches
are part of the proposal and may change during review.
