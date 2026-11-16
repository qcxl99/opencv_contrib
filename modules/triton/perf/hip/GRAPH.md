# Internal HIP graph metric

This optional helper measures an already warmed production `DeviceImage` call
captured repeatedly on one fixed nondefault stream. It does not implement a new
kernel, dispatch selector, public Graph owner, scheduler or primary benchmark.
It has no Torch dependency. `HipImage`/`HipStream` owning capture stays unsupported.

For an enabled HIP OpenCV module build, set `BUILD_PERF_TESTS=ON` and
`TRITON_BUILD_HIP_GRAPH_BENCH=ON` to build `opencv_perf_triton_hip_graph`. The option
is OFF by default and respects the selected perf module list. Inputs are private
benchmark requests like the existing native benchmark's JSON: `op`, fixed `case`,
`input_path`/`input_sha256`, `changed_path`/`changed_sha256`, `captured_calls` and
`samples`. Input files hold contiguous manifest-generated array bytes; only the
helper's independently allocated pitched buffers are passed to production calls.
No runtime pointer is accepted from the request. The request must declare safe
positive dimensions/depths and at most250calls; current timed samples are fixed10.

The caller must use its platform's approved idle-device/safety guards. A native
primitive control is `--primitive-control`; it captures a real64byteasyncmemset
and verifies three poisoned replays. First validate a tiny warmed production
request before increasing scope. Do not run archived binaries or infer another
platform's capture support. API/device errors stop the process without reset or
blind retries. The helper deliberately leaves unknown-completion resources to
supervised process exit instead of freeing potentially in-flight storage.

Before capture: allocate all input/output storage and canary margins; resolve the
actual public dispatch/JIT/table uploads; warm20calls and synchronize. Capture N
identical production calls, end capture, inspect nodes and instantiate. No helper
compile/module load/table upload may occur inside capture. Never choose a binary
by filename or bypass the real entry. The recorded production key/route and
verified artifact/manifest hashes name that path; graph kernel node parameters
record actual functions/grid/block/shared geometry and work counts.

Correctness checks include full ordinary-versus-graph equality, the existing
independent numerical reference, source mutation with newly computed reference,
two destination poison patterns, source/output canary bytes and multiple replays.
A CPU binding record rejects changed ptr/alignment/type/shape/pitch/parameters
before graph submission. The helper does not cache a graph across requests.
Each request also evicts the runtime LRU and replays while capture pins retain
modules/tables, then performs a second construct/replay/destroy cycle.

Timing: N=250productioncalls per graph for the fixed matrix,10timed replays.
Each sample is `hipEventElapsedTime(start,stop)/N` for one graph launch on the
same explicit stream; report raw total/per-call samples and median/p10/p90/range.
No minimum selection or L2 flush is inserted into the graph. Inputs/outputs are
reused, so this is a warm repeated-buffer graph metric, not ordinary caller
submission latency or isolated kernel latency. Host binding checks are outside
the timed bracket. The interval still includes graph launch/device scheduling
costs. Capture/instantiate/warmup/first production-call wall and ordinary250call
event/completion wall are separate observations. The raw field
`first_compile_call_wall_ms` measures the first production request in that process,
including conditional JIT/cache lookup/load; it is not isolated compiler time or
a promise every case was compiler-cold. Truly new artifact timestamps identify
actual cold builds; otherwise mark cache-warm lookup honestly.

Before freeing: synchronize successful work; destroy graphExec then graph; release
runtime capture pins only after all graphs are destroyed; reap completed launches;
then destroy events,free storage and destroy the stream. Input/output/stream and
context stay live throughout. Unknown completion never grants early release.

Results use the existing `graph_replay` metric spelling and preserve per-case raw
observations, not a new qualification schema. Without an identity-matched contract
baseline report `BASELINE_UNAVAILABLE`, candidate only; no graph ratio, primary
replacement, ordinary owning speedup or complete qualification claim. A graph
number never changes prior ordinary-call reports. Owning API support requires a
separate closed graph lifetime/transfer/completion contract; removing
`notCaptured()` is not such a contract.
