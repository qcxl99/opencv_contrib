Maintaining the Triton module {#tutorial_triton_maintenance}
=============================

The module builds and runs from this contrib tree without an external development
checkout or Torch. Kernel sources, compile-helper code and private C++ planning
sources are complete parts of this source package. They are not fetched or
regenerated from another repository during configuration. The runtime verifies
source and artifact content identity before trusting a JIT artifact.

License and algorithm sources
-----------------------------

The module contribution uses this repository's Apache-2.0 LICENSE; source files
carry the OpenCV project notice. These notices do not replace contributors'
responsibility to have the rights to submitted code. Preserve any original
third-party notices when incorporating further code. Triton and GPU SDKs are
external dependencies; their packages retain their own licenses.

The CPU-aligned algorithms and constants follow the selected OpenCV CPU
implementations in modules/imgproc/src/resize.cpp, color*.simd*.hpp and
imgwarp.cpp/warp dispatch code, under their applicable OpenCV source notices.
Resize reproduces the explicitly selected OpenCV 5.0.0 reference configuration;
cvtColor and warpAffine use the build-specific references documented by their APIs.

The module sources are maintained in this tree. Dispatch values in
src/runtime/dispatch_table.hpp are checked-in configuration, not a build-time
export from an external project. Keep their launch parameters consistent with
kernels/resize_cpu.py and the private provider. Record the actual OpenCV/contrib
revisions and local changes when producing validation evidence.

HIP kernels are derived with `python3 tools/make_kernels_cpu_hip.py`; the HIP
runtime is derived with `python3 tools/port_hip_runtime.py`. Both generators and
all their inputs live in this module. Run each with `--check` after editing the
corresponding CUDA source. A stale kernel derivation is rejected by configuration.
The resize derivation also includes kernels/resize_hip_fast.py, a self-contained
HIP-specific source. Preserve its layout/alignment guards in the provider.


Validation cases and numerical owners live under validation/. Common accuracy
and candidate perf sources use device.hpp; CMake selects device_cuda.cpp or
device_hip.cpp. These private files are not installed as a new public device API.
Keep SDK-specific lifecycle checks separate, preserve standard OpenCV filters,
and compare list/XML/executed counts. The backend-specific HIP executable exposes
five groups and reports its inner checks separately.

Use independent builds for module off, enabled accuracy/perf, samples and Python.
The selected compiler Python and the bindings interpreter have separate roles;
record both and verify the actual installed native extension and library paths.
Cold artifact-cache deployment, disk-cache use without Python and empty-cache
failure without a toolchain are separate checks. Do not label a warm compiler's
internal cache as a fully cold compiler measurement.

Performance XML/sanity files belong to a versioned run directory. Candidate-
generated sanity proves repeatability only; numerical accuracy needs the shared
independent reference. Public-call perf includes submission plus completion.
Device events, host issue, graph replay, startup and working-set observations
retain their own metric boundaries and raw samples. A missing baseline remains
a documented gap and cannot produce a speedup or full paired qualification.

The root README describes delivered interfaces and backend boundaries. See the
H20 guide for NVIDIA-specific first use; it does not grant HIP acceptance. The
BW1000 guide records the corresponding HIP commands and limitations.

See @ref tutorial_triton_validation for explicit CPU/source, module-off, enabled-build
and installation verification entry points and their separate evidence boundaries.
