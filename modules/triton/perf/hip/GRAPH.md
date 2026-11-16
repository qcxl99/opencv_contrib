# Internal HIP graph metric

This optional internal helper measures warmed production `DeviceImage` calls
captured repeatedly on one fixed nondefault stream. It uses the production
dispatch and kernels and has no Torch dependency.

For an enabled HIP OpenCV module build, set `BUILD_PERF_TESTS=ON` and
`TRITON_BUILD_HIP_GRAPH_BENCH=ON` to build `opencv_perf_triton_hip_graph`. The option
is OFF by default and respects the selected perf module list. Inputs are private
benchmark requests in JSON: `op`, `case`,
`input_path`/`input_sha256`, `changed_path`/`changed_sha256`, `captured_calls` and
`samples`. Input files hold contiguous array bytes identified by file hashes;
only the helper's independently allocated pitched buffers are passed to production calls.
No runtime pointer is accepted from the request. The request must declare safe
positive dimensions/depths and at most 250 captured calls; the number of timed
replays is fixed at 10.

Confirm the selected device is idle and follow the host's device-use requirements.
The `--primitive-control` option captures a 64-byte asynchronous memory fill and
checks three replays, overwriting the destination before each replay to verify
that the graph performs the expected writes. First validate a tiny warmed production
request before increasing scope. Do not run archived binaries or infer another
platform's capture support. API/device errors stop the process without reset or
blind retries. The helper deliberately leaves unknown-completion resources to
supervised process exit instead of freeing potentially in-flight storage.

Before capture: allocate input/output storage with surrounding guard bytes to
detect out-of-bounds writes; resolve the actual public dispatch, JIT compilation
and table uploads; execute 20 warmup calls and synchronize. Capture N
identical production calls, end capture, inspect nodes and instantiate. No helper
compile/module load/table upload may occur inside capture. Never choose a binary
by filename or bypass the real entry. The recorded production key/route and
verified hashes of the compiled binary and its metadata file identify that path;
graph kernel node parameters
record actual functions/grid/block/shared geometry and work counts.

Correctness checks include full ordinary-versus-graph equality, the existing
independent numerical reference, source mutation with newly computed reference,
two destination fill patterns, unchanged input/output guard bytes and multiple
replays. A host-side record of buffer addresses, alignment, types, shapes, row
pitches and operator parameters is checked before graph submission; mismatches
are rejected. The helper does not cache a graph across requests.
Each request also evicts entries from the least-recently-used runtime cache and
replays while retained references keep the graph's modules and tables alive,
then performs a second construct/replay/destroy cycle.

Timing: the fixed workload uses N = 250 production calls per graph and 10 timed
replays. Each sample is `hipEventElapsedTime(start,stop)/N` for one graph launch
on the same explicit stream; report raw total/per-call samples and median/p10/p90/range.
No minimum selection or L2 flush is inserted into the graph. Inputs/outputs are
reused, so this is a warm repeated-buffer graph metric, not ordinary caller
submission latency or isolated kernel latency. Host-side buffer and parameter
checks are outside the timed bracket. The interval still includes graph launch/device scheduling
costs. Host wall times for capture, instantiation, warmup and the first production
call are reported separately. Another measurement executes 250 ordinary calls
and reports both GPU-event time per call and host wall time per completed call. The raw field
`first_compile_call_wall_ms` measures the first production request in that process,
including conditional JIT/cache lookup/load. This is not isolated compiler time,
nor does it prove that compilation occurred. Use binary creation records to distinguish
new compilation from reuse of an existing compiled binary.

Before freeing: synchronize successful work; destroy graphExec then graph; release
the references retaining graph modules and tables only after all graphs are
destroyed; reclaim resources from completed launches;
then destroy events, free storage and destroy the stream. Input/output/stream and
context stay live throughout. Unknown completion never grants early release.

Results label the measurement `graph_replay` and preserve raw observations for
each case. This helper measures the Triton implementation only and does not run
a reference implementation for performance comparison. Its
`baseline_status: BASELINE_UNAVAILABLE` output therefore means that no baseline
latency or speedup ratio is available. Graph replay timings do not establish
ordinary-call performance or satisfy an application's correctness and performance
acceptance requirements.

Request validation is strict and runs before GPU allocation. Dimensions, channels,
budgets and ROI flags are checked without narrowing; strings, booleans and all
six matrix entries must have the stated types and finite values. Only the listed
mode aliases/borders are accepted. `roi` accepts a boolean or an exact
integer 0/1. Duplicate/unknown fields are refused. `case.id` is an optional string.
Any `case.borderValue` field is explicitly refused, including zero/null; this
exploratory tool currently measures the default zero border value only. Production
operator support is unchanged. Destination dimensions for colour conversion must
match its actual output dimensions. Requests cannot silently change matrix,
interpolation, shape or border settings; `request_sha256` identifies the exact
accepted JSON, including its explicit options and deterministic defaults.

This optional helper reports timings for the Triton implementation. CPU parser
tests do not establish HIP graph execution, and a result on one HIP architecture
does not establish support on another.

Run `tools/check_source.py` for request-validation and error-handling checks
without allocating GPU resources. Check actual graph execution on the intended
HIP device.

Same-size resize copies are explicitly refused by this graph tool because they
have no compiled-kernel manifest identity. Production same-size copy support is
unchanged. Float input files must contain finite values.
