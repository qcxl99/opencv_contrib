BW1000 first use {#tutorial_triton_bw1000_first_use}
================

This is a HIP procedure for Hygon BW1000/gfx936 with DTK and a HIP-capable
Triton compiler. It does not certify H20/CUDA. Keep the paired OpenCV/contrib
revisions and local patches, compiler identity and actual library paths with
each run. Never rebuild the pinned CPU oracle to make a test pass.

Select an idle physical device after checking occupancy. Use one visibility
mapping; stacking HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES can filter twice.
On this platform flagtree needs ROCM_PATH pointing to the DTK installation.
The bindings interpreter needs Python development headers and NumPy. The compile
helper interpreter must import HIP-capable Triton with Torch blocked; Torch is
not a dependency of the C++ module.

Example configuration (substitute the explicit source/build/install/interpreter
paths rather than copying another user's build cache):

```sh
cmake -S /path/to/opencv -B /path/to/build -G Ninja \
  -DOPENCV_EXTRA_MODULES_PATH=/path/to/opencv_contrib/modules \
  -DCMAKE_INSTALL_PREFIX=/path/to/install -DBUILD_opencv_triton=ON \
  -DBUILD_LIST=core,imgproc,triton,python3,ts \
  -DTRITON_BACKEND=hip -DWITH_CUDA=OFF -DTRITON_ROCM_PATH=/opt/dtk \
  -DTRITON_PYTHON_EXECUTABLE=/path/to/python -DTRITON_PYTHONPATH=/path/to/flagtree \
  -DPYTHON3_EXECUTABLE=/path/to/python -DBUILD_TESTS=ON \
  -DBUILD_PERF_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build /path/to/build --target opencv_triton opencv_test_triton \
  opencv_perf_triton opencv_test_triton_hip example_triton_device opencv_python3
cmake --install /path/to/build
```

Review CMake's actual module/dependency selection and interpreter paths. Ordinary
module-off builds require no Triton probe. Disabled BUILD_TESTS/PERF/EXAMPLES
must not leave the corresponding HIP targets enabled. The native Python extension
and all its loaded OpenCV libraries must come from the selected installation.
Set its library directory explicitly when relocating; the original install
RUNPATH may otherwise select transitive dependencies from the original tree.

In a selected device environment, run all three sample modes (`resize`, `warp`,
`cvtcolor`) and preserve the CPU-comparison results. Use a separately compiled
find_package consumer with installed public headers. Numerical Python GpuMat
adapters are unsupported on HIP: each of the three operations must raise a
catchable StsNotImplemented for omitted and None streams, never abort.

Use standard OpenCV list/filter/XML for opencv_test_triton. The supplemental
opencv_test_triton_hip has five registered groups; its checks attribute counts
internal assertions. Unknown or empty group selections fail before device work.
Compare planned inventory, actual filters, XML failures/skips and inner-check
counts. Common cases/reference/criteria owners are shared with CUDA; CUDA-only
context and driver tests are not silently counted as HIP execution.

Run opencv_perf_triton in a new versioned working directory. Write fresh sanity
once and verify it in another process with native OpenCV options. Candidate-
generated sanity is repeatability evidence, not the independent accuracy oracle.
Candidate perf cycles include public call submission plus caller-stream wait.
Native CUDA baseline targets are absent from a HIP-only installation. Preserve
this absence instead of producing a speedup from unmatched libraries.

For deployment, launch a new process with an empty OPENCV_TRITON_CACHE_DIR,
then another with the persisted cache and OPENCV_TRITON_PYTHON=/nonexistent.
The disk-hit call must succeed without a helper. A third, independent empty
cache with the same missing toolchain must fail explicitly without CPU fallback.
Record raw exit codes separately from expected outcomes. Empty module artifact
cache does not imply the compiler's internal cache was cleared or establish a
cold-start latency distribution.

Retain source/resource/binary hashes, actual commands, device occupancy and raw
samples. Host issue, HIP event span, graph replay, process startup and working
set measurements have different boundaries. A profiler trace is scoped to its
observed workload. Paired GPUfusion baseline qualification requires a verified
native HIP adapter and source/library isolation; the CUDA-only lab bridge must
not be relabeled as a HIP implementation. No H20 acceptance follows from these
checks. See maintenance.markdown and the module README for current scope.
