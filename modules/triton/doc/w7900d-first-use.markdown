W7900D first use {#tutorial_triton_w7900d_first_use}
================

This guide uses AMD Radeon PRO W7900D (`gfx1100`, wave32), ROCm 7.2.4 and a
ROCm-capable Triton 3.6.0 compiler. BW1000 uses `gfx936`, wave64 and DTK;
see @ref tutorial_triton_bw1000_first_use for that configuration. Both use the
HIP backend and C++ `DeviceImage` overloads. Do not transfer compiled kernels
or toolkit libraries between these platforms.

Configure and build
-------------------

Use the existing matching OpenCV and contrib sources, with build and install
directories outside the source tree. Select the existing Python environment
that imports the intended HIP Triton compiler. OpenCV bindings also need NumPy
and matching Python development headers and libraries. PyTorch is not required.

```sh
export ROCM_PATH=/opt/rocm
<compiler-python> -c "import sys; sys.modules['torch'] = None; import triton; print(triton.__version__)"
cmake -S <opencv> -B <build> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<install> \
  -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules \
  -DBUILD_LIST=core,imgproc,imgcodecs,triton,python3,python_bindings_generator,ts \
  -DBUILD_opencv_triton=ON -DTRITON_BACKEND=hip -DWITH_CUDA=OFF \
  -DTRITON_ROCM_PATH=/opt/rocm \
  -DTRITON_PYTHON_EXECUTABLE=<compiler-python> \
  -DTRITON_PYTHONPATH=<hip-triton-import-path> \
  -DPYTHON3_EXECUTABLE=<bindings-python> \
  -DCPU_BASELINE=SSE3 -DCPU_DISPATCH= \
  -DWITH_IPP=OFF -DWITH_OPENCL=OFF \
  -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON -DBUILD_EXAMPLES=ON \
  -DINSTALL_C_EXAMPLES=ON
cmake --build <build> --parallel 2
cmake --install <build>
```

Use the same ROCm installation for headers, linking and runtime loading. Check
the CMake summary and loaded libraries; `/opt/dtk` and Torch-bundled HIP libraries
are not substitutes for this configuration. The CPU options select the generic
resize reference configuration. Other CPU builds may differ numerically; see
@ref tutorial_triton_resize and @ref tutorial_triton_warpaffine.

Device selection and calls
-------------------------

Follow the server's device allocation and idle checks before running GPU work.
On the W7900D server, use only authorized physical devices 0 and 1, with one idle
device at a time. These commands assume physical device 0 is idle and reserved
for the task. Set visibility before loading any GPU runtime:

```sh
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export GPU_DEVICE_ORDINAL=0
```

ROCR first exposes physical device 0; the other selectors retain its logical
index 0. Confirm the actual PCI identity before execution. Visibility filters
limit compute selection, not driver initialization access to other render nodes.
Use the server's existing execution guard; these settings are not device isolation.
Do not reset devices or change the server's drivers to run these examples.

```sh
<build>/bin/example_triton_device resize
<build>/bin/example_triton_device cvtcolor
<build>/bin/example_triton_device warp
<build>/bin/opencv_test_triton --gtest_output=xml:accuracy.xml
<build>/bin/opencv_test_triton_hip --gtest_output=xml:hip.xml
<build>/bin/opencv_perf_triton --gtest_list_tests
```

Run commands serially with bounded timeouts and an idle-device check before
each job. Use OpenCV filters for the affected scope; list failures and skips.
The C++ calls use caller-owned storage and explicit native streams. Python can
query the backend, but numerical GpuMat calls on HIP raise `StsNotImplemented`;
there is no HIP Python device-buffer wrapper.

Runtime limits and deployment
-----------------------------

ROCm without `hipCtxGetId` uses retained primary-context identity. Without
`hipStreamGetCtx`, stream admission checks the device; arbitrary private-context
stream ownership is not established. Allocation checks verify device memory and
range, not arbitrary foreign-context provenance. These limits differ from DTK's
available context queries.

Check the installed headers and libraries with an independent C++ consumer.
Select the installed OpenCV Python path when checking backend queries and safe
refusals. For cache tests, installation and relocation, follow
@ref tutorial_triton_resize and @ref tutorial_triton_validation. A guide or an
older build's test result does not validate newly built artifacts.

The optional [HIP graph benchmark](../perf/hip/GRAPH.md) measures warmed replay
separately from ordinary calls. Performance comparisons need a named baseline
on this same device. Native ROCm ATen uses different arithmetic and requires a
separate process if its bundled HIP runtime differs; it is not the OpenCV
numerical reference or a dependency of this module.

Device AddressSanitizer requiring XNACK is unavailable for this gfx1100 setup.
Host sanitizers, output guards and lifecycle checks do not establish GPU sanitizer
coverage. Record unavailable checks explicitly rather than changing the driver
or reporting them as passed.
