Maintaining the Triton module {#tutorial_triton_maintenance}
=============================

The module builds and runs from this contrib tree without KernelForge or Torch.
The CUDA Python files under kernels/ and private C++ provider files are complete
vendored sources, with their original notices and recorded origin. They are part
of the module source package, not artifacts regenerated during OpenCV configure.
The runtime verifies their content identity before trusting a JIT artifact.

HIP kernels are derived with `python3 tools/make_kernels_cpu_hip.py`; the HIP
runtime is derived with `python3 tools/port_hip_runtime.py`. Both generators and
all their inputs live in this module. Run each with `--check` after editing the
corresponding CUDA source. A stale kernel derivation is rejected by configuration.
The resize derivation also includes kernels/resize_hip_fast.py, a self-contained
HIP-specific source. Preserve its layout/alignment guards in the provider.

The current research checkout additionally checks mirrored sources against
KernelForge using its export_module_kernels.py/test_module_sync.py. That external
synchronizer is a development integration check, not a user build dependency.
When maintaining exported module sources, reconcile an intentional change into
the mirrored lab source before invoking that exporter; do not blindly overwrite
module edits with a stale research checkout. The delivery includes the exact
module snapshot, related lab delta and source manifests so synchronization can
be reviewed. After upstream ownership changes, remove or update that mirror
agreement explicitly rather than presenting stale lab provenance as current.

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
