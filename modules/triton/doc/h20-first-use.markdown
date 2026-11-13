H20 first use {#tutorial_triton_h20_first_use}
=============

This guide uses Linux x86_64, NVIDIA H20 (`sm_90`), CUDA Toolkit 12.8 and Triton
3.6.0. Follow the [OpenCV Linux installation guide](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)
and the contrib repository build instructions for general prerequisites.
The module adds a runtime compiler dependency on Python and Triton; it does not
require PyTorch.

Configure and build
-------------------

Choose matching OpenCV and contrib sources. Keep build and install directories
outside the source tree. A Python environment with NumPy and matching Python
development headers is needed for bindings. The compiler interpreter can be
the same interpreter or another interpreter that imports Triton 3.6.0.

```sh
python3 -m venv <venv>
<venv>/bin/python -m pip install numpy triton==3.6.0
cmake -S <opencv> -B <build> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<install> \
  -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules \
  -DBUILD_LIST=core,imgproc,imgcodecs,triton,python3,python_bindings_generator,ts \
  -DBUILD_opencv_triton=ON -DTRITON_BACKEND=cuda -DWITH_CUDA=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 \
  -DCUDA_ARCH_BIN=9.0 -DCUDA_ARCH_PTX= \
  -DPYTHON3_EXECUTABLE=<venv>/bin/python \
  -DTRITON_PYTHON_EXECUTABLE=<venv>/bin/python \
  -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON \
  -DINSTALL_TESTS=ON
cmake --build <build> --parallel
cmake --install <build>
```

Check the CMake summary for the selected CUDA toolkit, device architecture,
Python headers/library and module list. `TRITON_PYTHONPATH` is optional if the
compiler interpreter already imports the intended Triton package. Kernels are
compiled on first use, rather than during the OpenCV build.

`cudawarping` and `cudaimgproc` are optional native CUDA performance comparisons.
Include them in `BUILD_LIST` to enable their available benchmark paths. They are
not production dependencies of the Triton module.

Use the installed module
------------------------

Select the installed OpenCV library and Python paths for the local installation
layout. Check `cv2.__file__`, the loaded native extension path and
`cv2.getBuildInformation()` to avoid importing another OpenCV installation.
Select an available device before running applications, for example with
`CUDA_VISIBLE_DEVICES`.

The examples use public GpuMat calls and explicit streams:

```sh
<build>/bin/example_triton_resize
<build>/bin/example_triton_cvtcolor
<build>/bin/example_triton_warpaffine
```

A Python pipeline can use the same operations:

```python
import cv2 as cv
import numpy as np

assert cv.triton.haveBackend()
stream = cv.cuda_Stream()
image = np.random.default_rng(7).integers(0, 256, (96, 128, 3), dtype=np.uint8)
src = cv.cuda_GpuMat()
src.upload(image, stream)
small = cv.triton.resize(src, (64, 48), interpolation=cv.INTER_LINEAR, stream=stream)
gray = cv.triton.cvtColor(small, cv.COLOR_BGR2GRAY, stream=stream)
matrix = np.eye(2, 3, dtype=np.float64)
warped = cv.triton.warpAffine(gray, matrix, (64, 48),
                             flags=cv.INTER_NEAREST, stream=stream)
stream.waitForCompletion()
result = warped.download()
```

Retain inputs and outputs until stream completion. For transparent borders,
initialize the destination before calling warpAffine. See
@ref tutorial_triton_warpaffine for reference arithmetic and accuracy bounds.

Tests and performance
---------------------

```sh
<build>/bin/opencv_test_triton --gtest_list_tests
<build>/bin/opencv_test_triton --gtest_output=xml:accuracy.xml
<venv>/bin/python <opencv_contrib>/modules/triton/misc/python/test/test_triton.py -v
<build>/bin/opencv_perf_triton --gtest_list_tests
```

The Python tests use OpenCV's test helpers; add
`<opencv>/modules/python/test` to `PYTHONPATH` as needed. Standard OpenCV
list/filter/XML options select test cases. Review actual failures and skips,
including tests requiring a second device. Performance workloads can be chosen
with `--gtest_filter`; warmed public-call timing includes submission and stream
completion. It is distinct from isolated kernel or graph replay timing.

Cache and installation
----------------------

Use a writable `OPENCV_TRITON_CACHE_DIR`. With a new directory, the first request
needs the selected Python/Triton compiler. Subsequent processes can use persisted
artifacts without an interpreter for matching requests. An empty cache without a
compiler must produce a clear toolchain error. An empty module artifact cache
does not clear the underlying compiler cache.

Installed resources are found relative to the module library. An external C++
consumer can use `find_package(OpenCV REQUIRED COMPONENTS core imgproc triton)`
and the installed public headers; it does not need private module headers.
When relocating the prefix, ensure the application and Python extension load
libraries from that prefix. See @ref tutorial_triton_resize for resource and
compiler overrides, and @ref tutorial_triton_maintenance for source generation.
