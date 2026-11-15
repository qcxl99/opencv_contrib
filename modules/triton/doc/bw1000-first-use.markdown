BW1000 first use {#tutorial_triton_bw1000_first_use}
================

This guide builds the HIP backend for Hygon BW1000/gfx936 with DTK 26.04 and
flagtree 3.6.0. The compiler interpreter must import the HIP-capable Triton
package. PyTorch is not a dependency. Set `ROCM_PATH` to the DTK installation
for both configuration and runtime compilation.

Build and install
-----------------

Replace the path placeholders with local paths. OpenCV Python bindings also
need matching Python development headers and NumPy.

```sh
export ROCM_PATH=/opt/dtk
cmake -S <opencv> -B <build> -G Ninja \
  -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules \
  -DCMAKE_INSTALL_PREFIX=<install> -DBUILD_opencv_triton=ON \
  -DBUILD_LIST=core,imgproc,triton,python3,ts \
  -DTRITON_BACKEND=hip -DWITH_CUDA=OFF -DTRITON_ROCM_PATH=/opt/dtk \
  -DTRITON_PYTHON_EXECUTABLE=<python-with-hip-triton> \
  -DTRITON_PYTHONPATH=<flagtree-package-directory> \
  -DPYTHON3_EXECUTABLE=<bindings-python> \
  -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON -DBUILD_EXAMPLES=ON \
  -DINSTALL_C_EXAMPLES=ON -DINSTALL_PYTHON_EXAMPLES=ON
cmake --build <build> --parallel
cmake --install <build>
```

Select an available device with the platform's visibility setting before
running an application. Avoid combining visibility filters that would filter
the selected devices twice. Device indices passed to the API refer to the
visible device set.

Use and test
------------

HIP Python applications use `cv2.triton.HipImage` and `cv2.triton.HipStream`.
The operators keep intermediate results on device; `upload()` and `download()`
provide blocking NumPy transfers. See @ref tutorial_triton_hip_python.

C++ applications can use the owning classes in `opencv2/triton/hip.hpp` or the
caller-owned `resizeDevice`, `cvtColorDevice` and `warpAffineDevice` entries in
`opencv2/triton/device.hpp`. The three modes of `example_triton_device` demonstrate
raw allocation, upload, native-stream submission, synchronization and CPU comparison:

```sh
<build>/bin/example_triton_device resize
<build>/bin/example_triton_device warp
<build>/bin/example_triton_device cvtcolor
<build>/bin/opencv_test_triton --gtest_output=xml:accuracy.xml
<build>/bin/opencv_test_triton_hip --gtest_output=xml:hip.xml
<build>/bin/opencv_perf_triton --gtest_list_tests
```

The standard accuracy and performance programs share cases and references with
the CUDA backend. CUDA-only tests and optional native CUDA timings are excluded
from HIP targets. The supplemental HIP executable groups backend-specific
checks; its XML records inner-check counts separately from registered tests.
Use OpenCV filters to choose performance workloads and report skipped tests.

Run the Python device pipeline and binding tests with the installed `cv2` on
`PYTHONPATH` and the selected installation's libraries on the loader path:

```sh
<bindings-python> <opencv_contrib>/modules/triton/samples/hip/device.py --device 0
<bindings-python> <opencv_contrib>/modules/triton/misc/python/test/test_triton_hip.py -v
```

The Python tests use OpenCV's test helpers; add
`<opencv>/modules/python/test` to `PYTHONPATH` as needed. The HIP numerical API
accepts `HipImage`, while GpuMat overloads raise `StsNotImplemented`.

Deployment
----------

Use installed public headers and `find_package(OpenCV)` for external consumers.
When relocating an installation, select its library and Python paths so that
an older OpenCV installation is not loaded accidentally.

A new `OPENCV_TRITON_CACHE_DIR` needs the compiler toolchain. Once artifacts are
cached, the same requests can run in another process without the compiler
interpreter. Python applications still require their Python runtime and `cv2`.
A separate empty cache with no compiler must raise a toolchain error. See
@ref tutorial_triton_resize for overrides and cache lifetime.

DTK permits allocation device/range and stream-context checks, but does not
support `HIP_POINTER_ATTRIBUTE_CONTEXT` for allocation-context queries. The
module does not claim to identify arbitrary foreign driver-context allocations
through that attribute.
