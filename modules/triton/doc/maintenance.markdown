Maintaining the Triton module {#tutorial_triton_maintenance}
=============================

@tableofcontents

Source layout
-------------

All sources and generation tools needed to build the module are in this
module directory. `kernels/` contains Triton implementations. `src/` contains
entry validation, adapters and runtime code. Runtime resize planning is separate
from the CPU output evaluator under `test/`; tests also compare against the
linked OpenCV CPU implementations where documented. Keep algorithmic source
attribution and license notices with the relevant implementation.

Describe current behavior in guides and explain invariants in code comments.
Keep development logs, run reports and build artifacts outside the source tree.
Generated-source notices and integrity hashes remain with their generated files.

Architecture and ownership {#triton_architecture}
--------------------------

The module provides explicit `cv::triton::*` calls. It does not intercept ordinary
`cv::*`, UMat or `cv::cuda::*` calls. Any future integration into main-library
accelerator dispatch is a separate design decision.

For a first source read, follow one call through these layers:

| Layer | Start here | Responsibility |
|---|---|---|
| Public API | `include/opencv2/triton.hpp`, `include/opencv2/triton/device.hpp` | Input/output types, supported options and asynchronous ownership |
| Entry validation | `src/cuda_resize_adapter.cpp`, `src/device_entries.cpp` | Output metadata, actual device views, overlap and stream admission |
| Operation plan | `src/runtime/provider.cpp`, `cvtcolor.cpp`, `warpaffine.cpp` | Sampling rules, kernel choice, compile key and launch shape |
| Backend runtime | `src/runtime/cuda_runtime.cpp` and its generated HIP counterpart | Compile/load, stream submission, caches and resource retention |
| Device computation | `kernels/` | The arithmetic and memory accesses executed on the GPU |
| Validation | `validation/`, `test/`, `perf/` | Shared cases, numerical references, lifecycle checks and timing |

`validation/` contains private shared accuracy/performance support. CMake
selects `device_cuda.cpp` or `device_hip.cpp`; each calls its backend's public
module entry points. These test helpers are not installed. Backend lifecycle
tests stay in their backend-specific source lists. Update common cases and
numerical criteria when adding support, and retain tests for unsupported inputs.

Size-dependent tables are built in C++, uploaded on the caller's stream and
cached. Cross-stream consumers wait on upload events. In-flight calls retain
modules, tables and upload staging bytes even after cache eviction. Warm calls
can be captured on a stream where compilation, module loads and table uploads
are already complete. After all graphs using these resources are destroyed or
cannot be replayed, call cv::triton::releaseGraphResources().

Asynchronous cleanup and context ownership
-----------------------------------------

An in-flight call retains its modules, device tables and upload host bytes until
completion is proven. The decisions in `src/runtime/lifetime.hpp` distinguish
unfinished work from failed cleanup:

| Condition | Cleanup behavior |
|---|---|
| Stream capture is active | Skip completion queries and reclamation, including failed-event metadata reaping. |
| Context is known destroyed | Release host ownership without querying old events or switching to that context. |
| Context state is unknown | Retain ownership and report an error; do not infer destruction. |
| Completion query succeeds | Release references for the completed operation and attempt event cleanup. |
| Completion query is not-ready | Retain resources for a later noncapturing check. A device-side wait alone does not prove upload host bytes can be released. |
| Completion query fails | Retain resources and report the error through submission/reap status, diagnostics and private counters. |
| Upload started without a valid recorded completion marker | Retain ownership until context destruction is established. |

If last-owner cleanup fails, a private process-lifetime retirement queue holds
the entire module/table owner, including upload host bytes. Noncapturing
submissions and explicit reaps retry eligible completion checks without adding
normal-path synchronization. An unknown result from free, unload or event
destruction is not retried as the same irreversible call; that owner remains
held until its context is known destroyed.

While failed resource cleanup remains unresolved, later submissions and explicit
cleanup calls report an error rather than silently accumulating resources.
Recovery requires proven completion/context destruction, or a process restart
for persistent driver failure. Inspect diagnostics and counters; retention until
process exit is not successful cleanup.

There is a distinct completed-event case: if event destruction fails after
completion was proven, module/table references may be released. The failure is
still reported, and the event handle/context identity remain as failed metadata
until context destruction is known or the process exits. That irreversible
destroy is not retried. No unfinished work is implied, so this metadata does not
block unrelated calls. `RuntimeStats::retired_resources` includes these handles.

Primary-context ownership follows these rules:

- The runtime retains one primary context per device for its process-lifetime caches.
- Host threads borrow it, including after `SetCurrent` failure; calls and short-lived
  threads do not add retains.
- HIP's `hipCtxGetId` path and its fallback share that owner. The fallback registry
  adds identity metadata, not another retain.
- Static teardown does not release the owner after the driver may be gone.
  Arbitrary primary-context resets and private contexts are unsupported by this fallback.

Runtime compiler, ABI and cache {#triton_runtime_toolchain}
-------------------------------

The module is disabled by default and requires no PyTorch dependency when
enabled. Nothing is compiled by Triton during the OpenCV build. On first use,
`tools/compile_helper.py` runs in a separate Python process using the configured
Triton 3.6.0 compiler. The C++ execution path does not embed an interpreter.
Build commands and toolkit examples are maintained in
@ref tutorial_triton_cuda_first_use and @ref tutorial_triton_hip_first_use.

The compile key includes source identity, kernel signature, compile-time
parameters, launch options, target and compiler version. The runtime checks
manifest and binary hashes before loading disk artifacts. An unsupported
compiler launch ABI is rejected. See @ref triton_architecture for table uploads,
stream dependencies and in-flight ownership.

The writable cache defaults to `~/.cache/opencv/triton`. The following runtime
settings override configured defaults:

| Setting | Purpose |
|---|---|
| `OPENCV_TRITON_CACHE_DIR` | Artifact cache directory |
| `OPENCV_TRITON_RESOURCES` | Kernel and helper resource directory |
| `OPENCV_TRITON_PYTHON` | Compiler interpreter |
| `OPENCV_TRITON_PYTHONPATH` | Compiler Python import path |

Installation places resources under `share/opencv5/triton`, resolved relative
to the module library. A disk-cache hit runs without the compiler interpreter;
a cache miss with no usable compiler raises a toolchain error.

The module and compiler have separate caches. Clearing
`OPENCV_TRITON_CACHE_DIR` forces a module-cache miss, but the compiler can still
reuse its own cached output. To measure compilation from scratch, also select
an empty compiler cache and record both cache locations.

Static installations use the configured resource install prefix when the
library-relative path is unavailable. Set `OPENCV_TRITON_RESOURCES` when relocating
a static SDK. Do not infer loaded-library identity from environment variables:
inspect the native extension and actual mapped OpenCV/CUDA/HIP libraries.

`TRITON_PYTHON_EXECUTABLE` and `TRITON_PYTHONPATH` configure the compiler helper;
`PYTHON3_EXECUTABLE` selects the OpenCV binding interpreter. They may differ.
`TRITON_BACKEND` selects CUDA or HIP; `TRITON_ROCM_PATH` configures the HIP toolkit
and the helper's default `ROCM_PATH`. Disabled-module builds do not probe the
module compiler. No toolchain or compiler package is installed by the module.

The supported launcher ABI is Triton 3.6.0, validated separately for CUDA cubin
and HIP hsaco metadata. Runtime argument types, ordered parameters, warp/block
geometry and zero scratch requirements are checked before launch. A matching
version string alone does not validate an incompatible backend or entry ABI.

Generated HIP sources
---------------------

Run the local generators from the module directory after changing their inputs:

```sh
python3 tools/make_kernels_cpu_hip.py
python3 tools/port_hip_runtime.py
python3 tools/make_kernels_cpu_hip.py --check
python3 tools/port_hip_runtime.py --check
```

The first generator derives HIP kernel sources, including the fast paths in
`kernels/resize_hip_fast.py` and `kernels/cvtcolor_hip_fast.py`. The second derives
the HIP runtime from `src/runtime/cuda_runtime.cpp` and the HIP-specific substitutions
in `tools/port_hip_runtime.py`. Edit these sources rather than generated files.
Preserve layout and alignment guards in the planners and runtime.
Configure checks reject stale generated files.

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

For deployment/cache checks, use @ref triton_runtime_toolchain and the backend
tutorials. Backend launch defaults and specialized layout guards are shared where
implemented; benchmark each actual device rather than infer per-architecture tuning.
