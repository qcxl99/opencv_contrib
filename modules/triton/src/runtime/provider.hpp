// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Private request and planning types shared by the module's operators.
// Not a public API or a cross-version plugin ABI. Adapters translate OpenCV/native types
// into device views; the plan resolves supported requests, tables, compile keys and grids.
// CUDA/HIP runtimes own actual memory, events, compilation, loading and submission.
// DeviceService/ResizeProvider below are private interface declarations, not the runtime's
// active dispatch mechanism. See doc/maintenance.markdown for source provenance.
// No OpenCV, CUDA, HIP or Torch type appears here.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "reference_cpu.hpp"

namespace kf_resize_rt {

// ------------------------------------------------------------------ identities

enum class Backend : int { CUDA = 1, HIP = 2 };
// U16 exists because warpAffine serves CV_16U. It is a property
// of this shared runtime, not of resize, which still accepts CV_8U and CV_32F only and says
// so at its own entry.
// S16 exists as an identity because CV_16S is a legal cv::warpAffine input and the type system
// has to be able to name it, but NO operator admits it yet: cv::warpAffine sends every CV_16S
// type down its legacy fixed-point path (imgwarp.cpp:2272, "// no default"), which is a
// different algorithm from the float kernels these operators reproduce. See warpaffine.cpp
// normalize() for the measurement.
enum class ElemType : int { U8 = 0, F32 = 1, U16 = 2, S16 = 3 };
inline std::size_t elem_size(ElemType t)
{
    return t == ElemType::U8 ? 1 : ((t == ElemType::U16 || t == ElemType::S16) ? 2 : 4);
}

// Which device context a buffer, stream or module belongs to. `native` is the raw
// CUcontext / hipCtx_t value; `generation` is obtained from the native context-lifetime
// identity, so a reused raw context handle cannot reuse another lifetime's resources. No cache keys a
// resource on a bare handle alone.
struct ContextId {
    Backend backend = Backend::CUDA;
    int device = -1;
    std::uintptr_t native = 0;
    std::uint64_t generation = 0;
    bool operator==(const ContextId& o) const {
        return backend == o.backend && device == o.device && native == o.native && generation == o.generation;
    }
};

// One device image. `ptr` points at the FIRST ELEMENT OF THE VIEW (the ROI start): the
// ROI offset has already been applied, exactly once, by the adapter. alloc_base/bytes
// describe the owning allocation, for range and overlap checks only; 0 = unknown.
struct DeviceView {
    ContextId ctx;
    std::uintptr_t ptr = 0;
    int rows = 0, cols = 0, channels = 0;
    ElemType type = ElemType::U8;
    std::size_t step = 0;               // row pitch in BYTES
    std::uintptr_t alloc_base = 0;
    std::size_t alloc_bytes = 0;
    std::size_t row_bytes() const { return static_cast<std::size_t>(cols) * channels * elem_size(type); }
    // [first byte, one past the last byte the view can touch)
    std::uintptr_t end() const { return rows <= 0 ? ptr : ptr + static_cast<std::size_t>(rows - 1) * step + row_bytes(); }
};

struct ExecContext {
    ContextId ctx;
    std::uintptr_t stream = 0;          // borrowed; the provider never creates a stream
};

// --------------------------------------------------------------------- status

enum class Code : int {
    Unsupported = 0,   // this request is not served; nothing was submitted, dst untouched
    Prepared = 1,      // plan, binary and device tables ready for enqueue
    Submitted = 2,     // work is on the stream; completes asynchronously
    Error = 3,         // compile / metadata / load / launch failure; see `submitted`
};

enum class Reason : int {
    None = 0,
    // Unsupported
    DepthOrChannels, Interpolation, EmptyOrSize, TypeMismatch, StepLayout, IndexRange,
    InPlaceOrOverlap, ContextMismatch,
    RouteNotEnabled,   // a valid request on a route this build does not serve yet
    // Error
    ToolchainUnavailable, CompileFailed, CompileTimeout, ManifestInvalid, LoadFailed,
    DeviceAlloc, LaunchFailed,
};
const char* reason_name(Reason r);

struct Status {
    Code code = Code::Unsupported;
    Reason reason = Reason::None;
    std::string message;
    // Error only: did any work reach the stream before the failure? After a submit the
    // caller must not assume dst is untouched, and must not silently re-run elsewhere.
    bool submitted = false;
    bool ok() const { return code == Code::Prepared || code == Code::Submitted; }
};

// ------------------------------------------------------------ request, plan

struct ResizeRequest {
    DeviceView src, dst;
    int dsize_w = 0, dsize_h = 0;       // > 0: dsize wins; else fx/fy
    double fx = 0, fy = 0;
    int interpolation = kf_resize_ref::INTER_LINEAR;
};

// The request after cv::resize parameter resolution. inv_x / inv_y are the CALLER'S
// scales when fx/fy were given (never re-derived from the rounded dsize).
struct Normalized {
    int src_h = 0, src_w = 0, dst_h = 0, dst_w = 0, channels = 0;
    ElemType type = ElemType::U8;
    int interpolation = 0;
    double inv_y = 0, inv_x = 0;
    kf_resize_ref::RouteInfo route;
};

// Pure checks, no device access. Unsupported carries the reason; the explicit API maps
// it to StsUnsupportedFormat / StsBadArg, a future dispatcher may try another provider.
//  * depth U8/F32, channels 1/3/4, interpolation NEAREST/LINEAR/CUBIC/AREA;
//  * src/dst same type and channel count, same context; dst size == resolved size;
//  * step a multiple of the element size and >= the row bytes;
//  * every scalar the kernels take fits int32 (rows, W = cols*cn, row strides in
//    elements, tap counts): beyond that the request is Unsupported, never truncated;
//  * no in-place and no overlap of the byte ranges [ptr, end()) (conservative).
// Source admission and output resolution before an adapter allocates dst. No device access.
Status resolve_output(const ResizeRequest& req, kf_resize_ref::Size2& out);
Status normalize(const ResizeRequest& req, Normalized& out);

// Shared layout/range validator; geometry and supported depth stay operator-owned.
Status validate_view(const DeviceView& view, const char* name);

// ------------------------------------------------------------ compile key

// What identifies a compiled binary, and nothing else. Triton is given no implicit
// specialisation (kernel.py: `cn: tl.constexpr`, every other runtime argument
// do_not_specialize), so this key is complete: two calls with equal keys run the same
// binary, whatever their sizes, strides or pointer alignment.
// `ry` is the RY constexpr of _resize_cpu_area_fast_kernel (window rows split over RY
// lanes) and is unused by every other kernel; 1 is the classic form and any value is
// bit-identical, so it only changes which binary is compiled.
//
// A FIELD LEFT AT 0 IS "NOT SUPPLIED": launch_shape() then takes that value from the
// dispatch table below, which mirrors _cpu_default / _CPU_TABLE in the exported resize kernel source.
// That is what the JIT path does, so a default-constructed LaunchConfig means "the JIT's
// own defaults for this request", not "BLOCK_H=1, BLOCK_W=256, 4 warps".
struct LaunchConfig { int block_h = 0, block_w = 0, num_warps = 0, num_stages = 3, ry = 0; };
struct Target { std::string backend = "cuda"; int arch = 90; int warp_size = 32; };
struct Toolchain {
    std::string compiler = "triton-3.6.0";      // compiler identity and version
    std::string source_sha256;                  // the shipped kernels source file
};

struct KernelKey {
    std::string kernel;                         // jit function name in the kernels source
    std::string signature;                      // runtime arguments, Triton type strings
    std::vector<std::pair<std::string, long>> constexprs;  // sorted by name
    LaunchConfig cfg;
    Target target;
    Toolchain toolchain;
    std::string text() const;                   // canonical, used for cache lookup and the manifest
};

// Key and grid for a normalized request; route copy has no kernel (is_copy, a device
// memcpy, reported as such and never as a Triton launch).
//
// cfg fields left at 0 are filled from default_config(n), the port of the dispatch table
// in kernel.py; an explicit value (as the compile protocol passes) is never overridden.
struct LaunchShape {
    bool is_copy = false;
    KernelKey key;
    unsigned grid[3] = {1, 1, 1};
};

// The launch shape _cpu_default(src_h, src_w, dst_h, dst_w, channels, mode, is_f32) picks
// for this request, as a LaunchConfig. num_stages is Triton's default (3) and `ry` applies
// only to ROUTE_AREA_FAST. tests/test_provider_protocol.py checks it against the JIT.
LaunchConfig default_config(const Normalized& n, const Target* target = nullptr);
LaunchShape launch_shape(const Normalized& n, const LaunchConfig& cfg, const Target& t, const Toolchain& tc,
                         int batch = 1, const DeviceView* src = nullptr, const DeviceView* dst = nullptr);

// ------------------------------------------------------- device service

// Private service interface declarations. Current CUDA/HIP runtimes perform these operations
// directly; these declarations do not prescribe a future OpenCV backend interface.
struct DeviceBuffer { ContextId ctx; std::uintptr_t ptr = 0; std::size_t bytes = 0; };
struct Event { std::uintptr_t handle = 0; };
struct Module { ContextId ctx; std::uintptr_t handle = 0; std::uintptr_t function = 0; int shared_bytes = 0; };

class DeviceService {
public:
    virtual ~DeviceService() = default;
    virtual Status current_context(int device, ContextId& out) = 0;
    virtual Status alloc(const ContextId& ctx, std::size_t bytes, DeviceBuffer& out) = 0;
    virtual void free(const DeviceBuffer& b) = 0;
    // `host` must stay valid until `done` has completed; the caller holds it in a lease.
    virtual Status upload_async(const ExecContext& ex, const DeviceBuffer& dst, const void* host,
                                std::size_t bytes, Event& done) = 0;
    virtual Status record(const ExecContext& ex, Event& out) = 0;
    virtual Status wait(const ExecContext& ex, const Event& e) = 0;   // stream waits on event
    virtual bool completed(const Event& e) = 0;
    virtual void release(const Event& e) = 0;
    virtual Status load(const ContextId& ctx, const std::vector<std::uint8_t>& binary,
                        const std::string& function, Module& out) = 0;
    virtual void unload(const Module& m) = 0;
    virtual Status launch(const ExecContext& ex, const Module& m, const unsigned grid[3],
                          unsigned threads, void** args) = 0;
    virtual Status copy2d_async(const ExecContext& ex, const DeviceView& dst, const DeviceView& src) = 0;
};

// ---------------------------------------------------------------- provider

// Resources a prepared call holds: the module, the device tables, the uploaded host
// copies. Shared ownership: a cache eviction only drops its index entry; the resource
// is freed when the last in-flight call holding it has completed (its event).
struct Lease;

struct Prepared {
    Normalized req;
    LaunchShape shape;
    std::vector<std::shared_ptr<Lease>> leases;
};

struct Support {
    bool device_ok = false;      // a backend and a usable context exist
    bool toolchain_ok = false;   // the compile helper can run (needed only on a cache miss)
    bool request_ok = false;     // normalize() accepts the request
    bool artifact_ready = false; // the binary for its key is already in a cache
    Status why;
};

class ResizeProvider {
public:
    virtual ~ResizeProvider() = default;
    // Pure query: no allocation, no submission, no compilation.
    virtual Support query(const ResizeRequest& req, const ExecContext& ex) = 0;
    // May compile (cold miss: the separate compile process), load, allocate and
    // upload tables on ex.stream. Never touches dst. Returns Prepared, Unsupported or Error.
    virtual Status prepare(const ResizeRequest& req, const ExecContext& ex, Prepared& out) = 0;
    // Submits on ex.stream, does not synchronise, records the completion event on every
    // lease. Returns Submitted or Error (with `submitted` set if any work reached the stream).
    virtual Status enqueue(Prepared& p, const ExecContext& ex) = 0;
};

}  // namespace kf_resize_rt
