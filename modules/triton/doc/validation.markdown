Building and checking a Triton contribution {#tutorial_triton_validation}
=========================================

@tableofcontents

These are explicit checks, not automatic first-use actions. Use the same paired
OpenCV/contrib source snapshot throughout and save commands, environment, source
changes, actual binaries and raw results. Run only the checks needed for the
change; a skipped hardware job is not a passed hardware job. This page defines
entry points, not new numerical or performance acceptance criteria.

CPU-only source checks
----------------------

On Linux with Python 3 and a C++17 compiler, from the contrib root:

```sh
python3 modules/triton/tools/check_source.py
```

This checks source license notices and Python syntax without importing kernels,
verifies both HIP generators with `--check`, then compiles and runs the pure C++
key/protocol controls in `test/cpu/test_protocol.cpp`. It needs neither OpenCV
binaries nor CUDA/HIP SDKs, Triton, Torch or a GPU. Controls cover request/key
round trips, source/target separation, response identity, incomplete artifacts,
validated scratch ABI, missing helpers and helper deadlines. This is not a
compiler integration or complete artifact publication/recovery test.

After an enabled OpenCV build, the CPU-only `Triton_Validation.GuardBandsBytewise`
gtest checks 8U/16U/32F, C1/C3/C4, unequal pitches and detection of byte corruption:

```sh
CUDA_VISIBLE_DEVICES= /path/to/build/bin/opencv_test_triton \
  --gtest_filter=Triton_Validation.GuardBandsBytewise \
  --gtest_output=xml:/path/to/new-results/guard.xml
```

The bytewise helper is shared by legacy CUDA ROI checks and common CUDA/HIP
fixtures. An expected guard image is constructed at the actual destination type;
16U data is never interpreted as float.

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

Enabled CUDA and HIP
--------------------

Use separate build/install directories, standard `BUILD_TESTS`,
`BUILD_PERF_TESTS`, `BUILD_EXAMPLES`, and the corresponding backend toolkit.
The @ref tutorial_triton_h20_first_use "H20 guide" contains the CUDA commands and installed
Python/consumer/cache checks. The @ref tutorial_triton_bw1000_first_use "BW1000 guide"
contains the HIP setup and device-entry checks. Include
`python3,python_bindings_generator,ts` in BUILD_LIST when validating wrappers
and tests; select compatible Python development headers/library and NumPy.

Both builds must compile the module, `opencv_test_triton`, `opencv_perf_triton`,
examples and `opencv_python3` when those options are enabled. HIP additionally
builds `opencv_test_triton_hip`. CUDA uses three GpuMat examples; HIP uses
`example_triton_device` in its three modes. Build all configured libraries before
installation, not just a sample that happens to link against an older library.

In a new result directory, use native discovery/filter/XML instead of hardcoded
historical counts. For example, with `BUILD_ENABLED` and `RESULTS` set explicitly:

```sh
mkdir -p "$RESULTS"
"$BUILD_ENABLED/bin/opencv_test_triton" --gtest_list_tests > "$RESULTS/accuracy-list.txt"
"$BUILD_ENABLED/bin/opencv_test_triton" --gtest_output="xml:$RESULTS/accuracy.xml"
"$BUILD_ENABLED/bin/opencv_perf_triton" --gtest_list_tests > "$RESULTS/perf-list.txt"
```

Run perf with the task's selected scope and OpenCV options from a fresh result
working directory. Perf smoke/execution is not performance qualification.
CUDA Python tests exercise the three real GpuMat wrappers. HIP Python tests
exercise backend queries and the explicitly unsupported GpuMat operations;
no HIP Python device-buffer wrapper is advertised. Separate positive, refusal,
platform skip, missing data and zero-test results.

Install and hardware evidence
-----------------------------

Verify `cv2.__file__`, actual loaded OpenCV libraries and runtime resources are
from the selected installation. Use installed headers and `find_package(OpenCV)`
for an independent C++ consumer. Check separate processes for a new artifact
cache, disk cache without Python, and a different empty cache without the
compiler. Existing runtime tests cover integrity and resource lifetime; select
changed boundaries, including graph capture or cross-stream dependencies when
applicable. First-use and deployment guides remain the command owners.

A hosted CPU source check, an enabled compile job and an actual H20/BW1000 run
establish different things. The repository's general PR workflow does not by
itself establish that this default-off module was enabled. Hardware runners and
integration into OpenCV's shared CI are to be agreed with maintainers; they are
not installed by this module. Pending/no checks on a pushed branch do not prove
failure or success. Until corresponding jobs exist, attach reproducible logs
with exact source/environment/binary identities and list each unrun backend.
Do not describe this document or a runner definition as an executed CI result.
