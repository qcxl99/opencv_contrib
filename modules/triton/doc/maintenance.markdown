Maintaining the Triton module {#tutorial_triton_maintenance}
=============================

Source layout
-------------

All sources and generation tools needed to build the module are in this
module directory. `kernels/` contains Triton implementations. `src/` contains
entry validation, adapters and runtime code. Runtime resize planning is separate
from the CPU output evaluator under `test/`; tests also compare against the
linked OpenCV CPU implementations where documented. Keep algorithmic source
attribution and license notices with the relevant implementation.

`validation/` contains private shared accuracy/performance support. CMake
selects `device_cuda.cpp` or `device_hip.cpp`; each calls its backend's public
module entry points. These test helpers are not installed. Backend lifecycle
tests stay in their backend-specific source lists. Update common cases and
numerical criteria when adding support, and retain tests for unsupported inputs.

Generated HIP sources
---------------------

Run the local generators from the module directory after changing their inputs:

```sh
python3 tools/make_kernels_cpu_hip.py
python3 tools/port_hip_runtime.py
python3 tools/make_kernels_cpu_hip.py --check
python3 tools/port_hip_runtime.py --check
```

The first generator derives HIP kernel sources, including the HIP resize fast
paths from `kernels/resize_hip_fast.py`. The second derives the HIP runtime from
`src/runtime/cuda_runtime.cpp`. Preserve layout and alignment guards in the
resize planner. Configure checks reject stale generated files.

Build and test
--------------

Use out-of-source builds. Check disabled-module builds and enable
`BUILD_TESTS`, `BUILD_PERF_TESTS`, `BUILD_EXAMPLES` and Python bindings as needed.
The Python used for bindings and the interpreter used by the runtime compiler
are selected independently. Use the intended installed extension and libraries
when testing an installation or a relocated prefix.

`opencv_test_triton` uses the common backend cases. CUDA builds add CUDA
lifecycle tests; HIP also provides `opencv_test_triton_hip`. Use the standard
OpenCV list/filter/XML options and review failures and skipped tests.

`opencv_perf_triton` measures warmed public calls with stream completion.
Generated performance sanity data checks repeatability; independent accuracy
tests check numerical behavior. Native CUDA comparison modules are optional
and do not alter production dependencies. Preserve comparable inputs, devices
and timing boundaries when reporting performance.

Check runtime compilation with a new `OPENCV_TRITON_CACHE_DIR`, then reuse the
persisted cache in a new process without a compiler interpreter. A separate
empty cache without a toolchain must fail clearly. Cache resources include
kernel sources and helper scripts, so changes to them require corresponding
compilation checks even when the C++ library is unchanged.

See @ref tutorial_triton_h20_first_use for CUDA commands and
@ref tutorial_triton_bw1000_first_use for HIP commands.

Static installations use the configured resource install directory when the
library-relative path is unavailable. Set `OPENCV_TRITON_RESOURCES` when moving
a static SDK to another prefix. Shared installations locate resources relative
to the library; both forms support the explicit override.
