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

Accuracy tests list private reference/protocol sources explicitly in CMake and
compile them as separate translation units. CUDA input-validation tests also compile
the private colour/warp plans into the test executable, without exporting those
functions from the module. Separate compilation is not an independent algorithmic
oracle: numerical independence still comes from the selected references and
checks. Keep each support source listed once per executable. Do not include a
`.cpp` file from another `.cpp`, or add an OBJECT library merely to hide duplicate
definitions. Check ordinary and world shared/static linking after changing this
boundary. The plan's no-contraction option is source-local, so a world build
does not change imgproc's floating-point compilation.

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

Asynchronous cleanup and context ownership
-----------------------------------------

Completion queries distinguish success, not-ready and error. Capture submission
skips event reclamation entirely. A query error is observable through a failed
runtime submission/reap, a diagnostic and private cleanup counters; the resource
is not released on an unknown completion status. Last-owner cleanup failures
retain the entire module/table owner, including upload host bytes, in a private
process-lifetime retirement queue. Noncapturing submissions and explicit noncapturing reaps
retry completion checks without synchronizing the normal path. An unknown outcome
of free/unload/event-destruction is not retried as the same irreversible call; that
owner remains held until its context is known destroyed. A successful device-side wait
alone does not release upload host memory. A failed/unrecorded completion marker
is retained until context destruction can be established; persistent driver
errors can therefore retain resources until process exit. Counters/diagnostics
must be inspected rather than treating this as successful cleanup. Known-destroyed
contexts release host ownership without querying old events or switching to them.
Unknown context errors retain ownership rather than guessing destruction.
While resources with failed cleanup remain held, later submissions and explicit
cleanup calls report an error instead of silently accumulating more resources; recovery requires proven
completion/context destruction, or process restart for persistent driver failure.

A runtime-created primary context has one deliberate process-lifetime retain per
device. Host threads borrow that owner, including after SetCurrent failure; they
do not add retains per invocation/thread. HIP context identification through `hipCtxGetId`, when available, and the
fallback that retains a primary context share the same resource owner. The fallback registry adds identity metadata, not
another retain. This keeps module/table caches valid. The module deliberately does
not release that owner during static teardown after the driver may be gone.
Arbitrary primary-context resets and private contexts are unsupported by this fallback.

If destruction of a completion event fails AFTER completion was proven, the
retained references to the completed operation's modules and tables may be released. The call still reports the failure;
its event handle/context identity are retained as failed metadata without retrying
that irreversible destroy. Metadata is dropped only when context destruction is
known, or at process exit. This completed-event case does not imply unfinished GPU
work and does not block later unrelated calls; the `RuntimeStats::retired_resources` counter includes
those retained event handles. Capture skips even this metadata reaping.
