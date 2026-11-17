Building and checking a Triton contribution {#tutorial_triton_validation}
=========================================

Choose a check: [source controls](#cpu-only-source-checks), [module off](#module-off),
[device builds](#enabled-cuda-and-hip), [installed applications](#installed-application-checks).

These are explicit checks, not automatic first-use actions. Use the same paired
OpenCV/contrib source snapshot throughout and save commands, environment, source
changes, actual binaries and raw results. Run only the checks needed for the
change; a skipped hardware job is not a passed hardware job. This page defines
entry points, not new numerical or performance acceptance criteria.

Private test linking and source style
------------------------------------

Changes to test-support source selection can be checked with independent build
directories for `BUILD_SHARED_LIBS=ON/OFF` and `BUILD_opencv_world=ON/OFF`, using
the enabled backend configuration below (include `world` in BUILD_LIST when
enabling it). Build `opencv_test_triton` and
`opencv_perf_triton`, then run the affected CPU-only controls with GPU visibility
disabled. A successful link is not GPU numerical or performance acceptance.

The source checker below does not check C++ style, include order or Doxygen.
Review handwritten code against neighbouring OpenCV code, preserve include
semantics, and inspect the diff after scoped formatting. Generated files should
be updated through their included generators and checked for consistency rather
than reformatted separately. Documentation generation is a separate check; state
whether it covers the module pages or the complete OpenCV website.

<a id="cpu-only-source-checks"></a>
CPU-only source checks
----------------------

On Linux with Python 3 and a C++17 compiler, from the contrib root:

```sh
python3 modules/triton/tools/check_source.py
```

This checks source license notices and Python syntax without importing kernels,
verifies both HIP generators with `--check`, and runs the CPU controls for the
compile protocol, graph requests and resource lifetime under `test/cpu/`.
Native context/cleanup code is compiled with injected driver calls.
It needs neither OpenCV
binaries nor CUDA/HIP SDKs, Triton, Torch or a GPU. Controls cover request/key
round trips, source/target separation, response identity, incomplete artifacts,
validated scratch ABI, missing helpers and helper deadlines. Graph controls
check request admission; lifetime controls check capture, completion failures
and context ownership without device work. These are not GPU execution or
compiler-integration checks.

After an enabled OpenCV build, the CPU-only `Triton_Validation.GuardBandsBytewise`
gtest checks 8U/16U/32F, C1/C3/C4, unequal pitches and detection of byte corruption:

```sh
CUDA_VISIBLE_DEVICES= /path/to/build/bin/opencv_test_triton \
  --gtest_filter=Triton_Validation.GuardBandsBytewise \
  --gtest_output=xml:/path/to/new-results/guard.xml
```

The bytewise helper is shared by CUDA ROI checks and common CUDA/HIP
fixtures. An expected guard image is constructed at the actual destination type;
16U data is never interpreted as float.

<a id="module-off"></a>
Module off
----------

Use a new out-of-source directory. Here `OPENCV_SRC`, `CONTRIB_SRC` and `BUILD_OFF`
are absolute paths selected by the caller:

```sh
cmake -S "$OPENCV_SRC" -B "$BUILD_OFF" -G Ninja \
  -DOPENCV_EXTRA_MODULES_PATH="$CONTRIB_SRC/modules" \
  -DBUILD_LIST=core,imgproc,triton -DBUILD_opencv_triton=OFF \
  -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF
cmake --build "$BUILD_OFF" --parallel 4
```

Check CMake's selected modules/targets: Triton must be absent. It must not probe
the module's compiler Python or require a CUDA/HIP/Triton/Torch installation.
OpenCV itself may discover Python for other build purposes; that is distinct
from this module introducing a dependency. To verify absence of a module probe,
set `TRITON_PYTHON_EXECUTABLE` to a logging stub in this off-only build and verify
it was never invoked. Do not use that stub in enabled builds.

<a id="enabled-cuda-and-hip"></a>
Enabled CUDA and HIP
--------------------

Use separate build/install directories, standard `BUILD_TESTS`,
`BUILD_PERF_TESTS`, `BUILD_EXAMPLES`, and the corresponding backend toolkit.
Use [CUDA first use][cuda-first-use] for CUDA build/install commands and
[HIP first use][hip-first-use] for HIP, including DTK/ROCm differences.
Results from one machine or architecture do not validate another. Include
`python3,python_bindings_generator,ts` in BUILD_LIST when validating wrappers
and tests; select compatible Python development headers/library and NumPy.

Both builds must compile the module, `opencv_test_triton`, `opencv_perf_triton`,
examples and `opencv_python3` when those options are enabled. HIP additionally
builds `opencv_test_triton_hip`. CUDA uses three GpuMat examples; HIP uses
`example_triton_device` in its three modes. Build all configured libraries before
installation, not just a sample that happens to link against an older library.

In a new result directory, use native discovery/filter/XML instead of hardcoded
stored counts. For example, with `BUILD_ENABLED` and `RESULTS` set explicitly:

```sh
mkdir -p "$RESULTS"
"$BUILD_ENABLED/bin/opencv_test_triton" --gtest_list_tests > "$RESULTS/accuracy-list.txt"
"$BUILD_ENABLED/bin/opencv_test_triton" --gtest_output="xml:$RESULTS/accuracy.xml"
"$BUILD_ENABLED/bin/opencv_perf_triton" --gtest_list_tests > "$RESULTS/perf-list.txt"
```

Run perf with the task's selected scope and OpenCV options from a fresh result
working directory. A successful perf run alone does not demonstrate correctness
or attainment of an application's performance requirements.
CUDA Python tests exercise the three real GpuMat wrappers. HIP Python tests
exercise backend queries and the explicitly unsupported GpuMat operations;
no HIP Python device-buffer wrapper is advertised. Separate positive, refusal,
platform skip, missing data and zero-test results.

<a id="installed-application-checks"></a>
Installed application checks
----------------------------

Verify `cv2.__file__`, actual loaded OpenCV libraries and runtime resources are
from the selected installation. Use installed headers and `find_package(OpenCV)`
for an independent C++ consumer. Check separate processes for a new artifact
cache, disk cache without a compiler interpreter, and a different empty cache
without the compiler. Existing runtime tests cover integrity and resource lifetime; select
changed boundaries, including graph capture or cross-stream dependencies when
applicable. Backend tutorials own build/install commands;
[runtime compiler/cache guide][runtime-toolchain] owns compiler, cache and relocation rules.

Report the actual device, toolchain and source/installed-binary identity with
results. List failures and skips separately. CPU controls, build success and
hardware execution cover different boundaries; compare each result with the
operation and backend it exercised.

<!-- Link targets: GitHub source paths; Doxygen page labels in the hidden block. -->
<!--
@cond
-->
[cuda-first-use]: ../tutorials/cuda-first-use.markdown
[hip-first-use]: ../tutorials/hip-first-use.markdown
[runtime-toolchain]: maintenance.markdown#runtime-toolchain
<!--
@endcond
[cuda-first-use]: @ref tutorial_triton_cuda_first_use
[hip-first-use]: @ref tutorial_triton_hip_first_use
[runtime-toolchain]: @ref triton_runtime_toolchain
@cond
-->
<!-- @endcond -->
