# Private validation support

This directory is not installed and is not a public DeviceMat/Stream API.
`device.hpp` is the small device interface used by common accuracy and candidate
perf sources. `device_cuda.cpp` owns only test resources and calls the real
GpuMat-based `cv::triton::*` entries; it never calls the private runtime directly.
Host upload/download are blocking setup/result operations outside perf cycles.
Candidate perf cycles include public submission and caller-stream completion;
they are synchronized call timings, not pure kernel or graph replay timings.

`reference.hpp`, `resize_cases.hpp` and `cvtcolor_cases.hpp` share existing
reference rules, numerical bounds and independent case specifications with the
legacy CUDA tests. Common coverage supplements existing tests; native context,
graph, deterministic blocked-stream retention and CUDA-specific boundary tests
remain in their original files. No H20 result qualifies another target.

For another backend, supply one implementation of `device.hpp` against its
actual public entry and select it in CMake instead of `device_cuda.cpp`.
Preserve parent/ROI lifetime, row pitch and output reuse. `Stream::waitFor`
means enqueue a dependency on the producer's current work, not synchronously
wait on the host. Source tests and perf remain shared; backend-only tests stay
in the adapter-specific source list. Keep the standard OpenCV test/perf targets
and build switches. Do not copy the references, tolerances or case tables.

The production boundary is `src/entry.hpp` plus existing private provider
requests (`DeviceView`, `ContextId`, `ExecContext`). Adapters verify native
ownership, obtain output metadata, allocate/reuse through the owning framework,
then re-read and validate the destination before submission. Runtime still owns
JIT caches and internal resources. UMat/non-CPU HAL can supply a future adapter;
no new public memory manager or upstream registration ABI is prescribed here.

CMake selects device_cuda.cpp for CUDA and device_hip.cpp for HIP. HIP calls
public device-buffer entries, including a scale-aware resize overload that
preserves the original fx/fy values. Both implementations share cases and
reference/criteria owners. validation/numerics.hpp is also reused by the
backend-specific HIP lifecycle groups. No validation header is installed.

The HIP adapter preserves parent allocation lifetime in ROI views and enqueues
event dependencies for waitFor. Setup/result transfers are synchronous. Standard
accuracy and perf targets obey BUILD_TESTS/BUILD_PERF_TESTS; CUDA-only lifecycle
sources and native CUDA baselines are excluded on HIP. The separate
opencv_test_triton_hip exposes five registered group tests: list, positive and
negative glob filters and XML refer to those groups; inner checks are recorded
separately, not presented as independently registered gtests.

On DTK, HIP_POINTER_ATTRIBUTE_CONTEXT is unsupported. The production device
entry checks actual allocation device/range and obtains active context/generation
and stream context through supported queries. This is not evidence that arbitrary
foreign driver-context allocation ownership can be identified on DTK.
Acceptance remains backend-specific; H20 evidence does not qualify BW1000.
