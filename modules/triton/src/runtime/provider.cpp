// Pure parts of the provider protocol: request normalisation and validation, the explicit
// compile key, launch geometry. No device access. See provider.hpp.
#include "provider.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "dispatch_table.hpp"

namespace kf_resize_rt {

namespace ref = kf_resize_ref;

const char* reason_name(Reason r) {
    switch (r) {
        case Reason::None: return "none";
        case Reason::DepthOrChannels: return "depth_or_channels";
        case Reason::Interpolation: return "interpolation";
        case Reason::EmptyOrSize: return "empty_or_size";
        case Reason::TypeMismatch: return "type_mismatch";
        case Reason::StepLayout: return "step_layout";
        case Reason::IndexRange: return "index_range";
        case Reason::InPlaceOrOverlap: return "in_place_or_overlap";
        case Reason::ContextMismatch: return "context_mismatch";
        case Reason::RouteNotEnabled: return "route_not_enabled";
        case Reason::ToolchainUnavailable: return "toolchain_unavailable";
        case Reason::CompileFailed: return "compile_failed";
        case Reason::CompileTimeout: return "compile_timeout";
        case Reason::ManifestInvalid: return "manifest_invalid";
        case Reason::LoadFailed: return "load_failed";
        case Reason::DeviceAlloc: return "device_alloc";
        case Reason::LaunchFailed: return "launch_failed";
    }
    return "?";
}

namespace {

Status prepared() {
    Status s;
    s.code = Code::Prepared;
    return s;
}

Status unsupported(Reason r, std::string msg) {
    Status s;
    s.code = Code::Unsupported;
    s.reason = r;
    s.message = std::move(msg);
    return s;
}

constexpr long long I32_MAX = std::numeric_limits<int>::max();

bool fits_i32(long long v) { return v >= 0 && v <= I32_MAX; }

Status check_view(const DeviceView& v, const char* which) {
    if (v.ptr == 0 || v.rows <= 0 || v.cols <= 0) return unsupported(Reason::EmptyOrSize, std::string(which) + " is empty");
    const std::size_t es = elem_size(v.type);
    if (v.step % es != 0 || v.step < v.row_bytes())
        return unsupported(Reason::StepLayout, std::string(which) + ": step must be a multiple of the element size and >= the row bytes");
    if (!fits_i32(static_cast<long long>(v.step / es)) || !fits_i32(static_cast<long long>(v.cols) * v.channels))
        return unsupported(Reason::IndexRange, std::string(which) + ": row stride or row width exceeds int32 elements");
    if (v.alloc_base != 0 && (v.ptr < v.alloc_base || v.end() > v.alloc_base + v.alloc_bytes))
        return unsupported(Reason::StepLayout, std::string(which) + " extends beyond its allocation");
    return prepared();
}

const char* tname(ElemType t) { return t == ElemType::U8 ? "u8" : "fp32"; }

// The runtime-argument lists of the four jit functions in operators/resize/kernel.py,
// in Triton's type strings. Constexpr parameters are not runtime arguments. Every
// integer is i32 (normalize() keeps them in range), so Triton's value-dependent
// i32/i64 typing never applies. tests/test_provider_protocol.py checks these strings
// against the signatures the JIT actually compiles.
std::string signature(ref::Route r, ElemType t) {
    const std::string T = tname(t);
    const std::string io = "src:*" + T + ", dst:*" + T + ", ";
    const std::string tail = "src_rs:i32, dst_rs:i32, src_bs:i32, dst_bs:i32";
    switch (r) {
        case ref::ROUTE_NEAREST:
            return io + "xofs:*i32, yofs:*i32, dst_h:i32, W:i32, " + tail;
        case ref::ROUTE_LINEAR:
        case ref::ROUTE_CUBIC: {
            const std::string w = t == ElemType::U8 ? "i32" : "fp32";  // fixed-point vs float coefficients
            return io + "xofs:*i32, ax:*" + w + ", yofs:*i32, by:*" + w + ", byf:*fp32, "
                        "src_h:i32, src_w:i32, dst_h:i32, W:i32, xmax:i32, " + tail;
        }
        case ref::ROUTE_AREA_FAST:
            return io + "src_h:i32, src_w:i32, dst_h:i32, W:i32, dwidth1:i32, scale:fp32, IX:i32, IY:i32, " + tail;
        case ref::ROUTE_AREA:
            return io + "xs:*i32, xa:*fp32, xc:*i32, ys:*i32, ya:*fp32, yc:*i32, KX:i32, KY:i32, dst_h:i32, W:i32, " + tail;
        case ref::ROUTE_COPY:
            break;
    }
    return "";
}

}  // namespace

Status normalize(const ResizeRequest& req, Normalized& out) {
    const DeviceView &s = req.src, &d = req.dst;
    if (req.interpolation < ref::INTER_NEAREST || req.interpolation > ref::INTER_AREA)
        return unsupported(Reason::Interpolation, "interpolation must be NEAREST, LINEAR, CUBIC or AREA");
    for (const DeviceView* v : {&s, &d})
        if (v->channels != 1 && v->channels != 3 && v->channels != 4)
            return unsupported(Reason::DepthOrChannels, "channels must be 1, 3 or 4");
    if (s.type != d.type || s.channels != d.channels)
        return unsupported(Reason::TypeMismatch, "src and dst must have the same depth and channel count");
    if (!(s.ctx == d.ctx)) return unsupported(Reason::ContextMismatch, "src and dst belong to different contexts");
    Status st = check_view(s, "src");
    if (!st.ok()) return st;
    st = check_view(d, "dst");
    if (!st.ok()) return st;

    ref::Size2 z;
    try {
        z = ref::resolve_size(s.rows, s.cols, req.dsize_w, req.dsize_h, req.fx, req.fy);
    } catch (const std::invalid_argument& e) {
        return unsupported(Reason::EmptyOrSize, e.what());
    }
    if (z.dst_h <= 0 || z.dst_w <= 0) return unsupported(Reason::EmptyOrSize, "resolved size is empty");
    if (d.rows != z.dst_h || d.cols != z.dst_w)
        return unsupported(Reason::EmptyOrSize, "dst size differs from the resolved size");
    if (s.ptr < d.end() && d.ptr < s.end())
        return unsupported(Reason::InPlaceOrOverlap, "src and dst overlap (in-place is not supported)");

    out.src_h = s.rows; out.src_w = s.cols; out.dst_h = z.dst_h; out.dst_w = z.dst_w;
    out.channels = s.channels; out.type = s.type; out.interpolation = req.interpolation;
    out.inv_y = z.inv_y; out.inv_x = z.inv_x;
    out.route = ref::route(out.src_h, out.src_w, out.dst_h, out.dst_w, out.inv_y, out.inv_x, out.interpolation);
    if (out.route.kind == ref::ROUTE_AREA) {  // tap counts are kernel scalars too
        const ref::AreaTab tx = ref::area_tab(out.src_w, out.dst_w, 1.0 / out.inv_x);
        const ref::AreaTab ty = ref::area_tab(out.src_h, out.dst_h, 1.0 / out.inv_y);
        if (!fits_i32(static_cast<long long>(tx.taps) * out.dst_w) || !fits_i32(static_cast<long long>(ty.taps) * out.dst_h))
            return unsupported(Reason::IndexRange, "area tap tables exceed int32 indexing");
    }
    return prepared();
}

std::string KernelKey::text() const {
    std::ostringstream o;
    o << "kernel=" << kernel << ";sig=" << signature << ";cx=";
    for (std::size_t i = 0; i < constexprs.size(); ++i)
        o << (i ? "," : "") << constexprs[i].first << "=" << constexprs[i].second;
    o << ";nw=" << cfg.num_warps << ";ns=" << cfg.num_stages << ";target=" << target.backend << ":" << target.arch
      << ":" << target.warp_size << ";compiler=" << toolchain.compiler << ";source=" << toolchain.source_sha256;
    return o.str();
}

LaunchConfig default_config(const Normalized& n) {
    LaunchConfig cfg;
    cfg.num_stages = DISPATCH_DEFAULT_NUM_STAGES;
    // The class comes from the route the request actually takes (n.route, already resolved
    // from the caller's fx/fy by normalize()). Re-deriving it from the rounded sizes would
    // pick a different table row for fx/fy requests: fx = 1/3 runs area_fast while its
    // rounded sizes look like area, and the two rows differ. This mirrors _cpu_default,
    // which takes the caller's inverses for the same reason; tests/test_provider_protocol.py
    // compares the two keys byte for byte.
    const ref::RouteInfo look = n.route;
    const char* want = ref::route_name(look.kind);
    const char* ydir = n.dst_h > n.src_h ? "up" : "down";
    const bool two = look.kind == ref::ROUTE_AREA_FAST && look.ix == 2 && look.iy == 2;
    const bool f32 = n.type == ElemType::F32;
    // The (route, cn, dtype, ydir[, two_by_two]) row; for area_fast a miss falls back to the
    // same row with two_by_two = false. Only "down" rows exist, and the index is
    // (route, cn, is_f32, yd) -- an absent "up" row is a miss, not a down row.
    auto match = [&](bool two_by_two) -> const DispatchRow* {
        for (const DispatchRow& r : DISPATCH_TABLE)
            if (std::strcmp(r.route, want) == 0 && r.cn == n.channels && r.f32 == f32 &&
                std::strcmp(r.ydir, ydir) == 0 && r.two_by_two == two_by_two)
                return &r;
        return nullptr;
    };
    const DispatchRow* row = match(two);
    if (!row && look.kind == ref::ROUTE_AREA_FAST) row = match(false);
    if (row) {
        cfg.block_h = row->block_h; cfg.block_w = row->block_w; cfg.num_warps = row->num_warps;
    } else {
        cfg.block_h = DISPATCH_FALLBACK_BLOCK_H;
        cfg.block_w = DISPATCH_FALLBACK_BLOCK_W;
        cfg.num_warps = DISPATCH_FALLBACK_NUM_WARPS;
    }
    // _cpu_parallel: shrink BLOCK_H, then BLOCK_W, until the launch has at least
    // DISPATCH_MIN_PROGRAMS programs (or neither can shrink). It counts programs on
    // dst_w for axis 1 whatever the route; that only errs on the small side for C3/C4
    // element-major tiles, which is the safe direction. A copy keeps CPU_CONFIG untouched.
    if (look.kind != ref::ROUTE_COPY) {
        auto programs = [&](const LaunchConfig& c) {
            return static_cast<long long>((n.dst_h + c.block_h - 1) / c.block_h) *
                   ((n.dst_w + c.block_w - 1) / c.block_w);
        };
        while (programs(cfg) < DISPATCH_MIN_PROGRAMS && cfg.block_h > 1) cfg.block_h = std::max(1, cfg.block_h / 2);
        while (programs(cfg) < DISPATCH_MIN_PROGRAMS && cfg.block_w > 64) cfg.block_w = std::max(64, cfg.block_w / 2);
    }
    // RY is only a constexpr of the area_fast kernel; the JIT passes cfg.get("RY", 1).
    cfg.ry = 1;
    return cfg;
}

LaunchShape launch_shape(const Normalized& n, const LaunchConfig& requested, const Target& t, const Toolchain& tc,
                         int batch) {
    LaunchShape ls;
    const ref::Route r = n.route.kind;
    if (r == ref::ROUTE_COPY) { ls.is_copy = true; return ls; }
    // Fields the caller left at 0 come from the JIT's dispatch table. An explicitly supplied
    // value is never replaced, so the compile protocol's (block_h, block_w, num_warps) pass
    // through unchanged.
    LaunchConfig cfg = requested;
    const LaunchConfig d = default_config(n);
    if (cfg.block_h <= 0) cfg.block_h = d.block_h;
    if (cfg.block_w <= 0) cfg.block_w = d.block_w;
    if (cfg.num_warps <= 0) cfg.num_warps = d.num_warps;
    if (cfg.num_stages <= 0) cfg.num_stages = d.num_stages;
    if (cfg.ry <= 0) cfg.ry = d.ry;
    KernelKey& k = ls.key;
    k.cfg = cfg; k.target = t; k.toolchain = tc;
    k.signature = signature(r, n.type);
    const long u8 = n.type == ElemType::U8 ? 1 : 0;
    std::vector<std::pair<std::string, long>>& c = k.constexprs;
    c = {{"BLOCK_H", cfg.block_h}, {"BLOCK_W", cfg.block_w}, {"cn", n.channels}};
    switch (r) {
        case ref::ROUTE_NEAREST: k.kernel = "_resize_cpu_nearest_kernel"; break;
        case ref::ROUTE_LINEAR:
        case ref::ROUTE_CUBIC:
            k.kernel = "_resize_cpu_sep_kernel";
            c.push_back({"KSIZE", r == ref::ROUTE_CUBIC ? 4 : 2});
            c.push_back({"U8", u8});
            break;
        case ref::ROUTE_AREA_FAST:
            k.kernel = "_resize_cpu_area_fast_kernel";
            c.push_back({"RY", cfg.ry});
            c.push_back({"TWO_BY_TWO", n.route.ix == 2 && n.route.iy == 2 ? 1 : 0});
            c.push_back({"U8", u8});
            break;
        case ref::ROUTE_AREA:
            k.kernel = "_resize_cpu_area_kernel";
            c.push_back({"U8", u8});
            break;
        case ref::ROUTE_COPY: break;
    }
    std::sort(c.begin(), c.end());
    // Grid axis 1 counts the tile's second dimension, which the kernels agree on with
    // _cpu_launch_spec: the pixel-major area kernels tile BLOCK_W output PIXELS, the
    // element-major (row-sliding) nearest and separable ones tile BLOCK_W channel values.
    const long long W = static_cast<long long>(n.dst_w) * n.channels;
    const long long axis1 = (r == ref::ROUTE_AREA || r == ref::ROUTE_AREA_FAST) ? n.dst_w : W;
    ls.grid[0] = static_cast<unsigned>((n.dst_h + cfg.block_h - 1) / cfg.block_h);
    ls.grid[1] = static_cast<unsigned>((axis1 + cfg.block_w - 1) / cfg.block_w);
    ls.grid[2] = static_cast<unsigned>(batch);
    return ls;
}

}  // namespace kf_resize_rt

// ---------------------------------------------------------------- C ABI (tests)

namespace {
kf_resize_rt::DeviceView view(std::uintptr_t ptr, int rows, int cols, int cn, int is_f32, std::size_t step,
                              std::uintptr_t ctx) {
    kf_resize_rt::DeviceView v;
    v.ctx.native = ctx; v.ctx.device = 0;
    v.ptr = ptr; v.rows = rows; v.cols = cols; v.channels = cn;
    v.type = is_f32 ? kf_resize_rt::ElemType::F32 : kf_resize_rt::ElemType::U8;
    v.step = step;
    return v;
}
}  // namespace

// normalize() over raw fields. Returns the Code; *reason gets the Reason; *route the route.
extern "C" int kf_rt_normalize(std::uintptr_t src_ptr, int src_rows, int src_cols, std::size_t src_step, std::uintptr_t src_ctx,
                               std::uintptr_t dst_ptr, int dst_rows, int dst_cols, std::size_t dst_step, std::uintptr_t dst_ctx,
                               int src_cn, int dst_cn, int src_f32, int dst_f32, int dsize_w, int dsize_h, double fx, double fy,
                               int interp, int* reason, int* route) {
    kf_resize_rt::ResizeRequest q;
    q.src = view(src_ptr, src_rows, src_cols, src_cn, src_f32, src_step, src_ctx);
    q.dst = view(dst_ptr, dst_rows, dst_cols, dst_cn, dst_f32, dst_step, dst_ctx);
    q.dsize_w = dsize_w; q.dsize_h = dsize_h; q.fx = fx; q.fy = fy; q.interpolation = interp;
    kf_resize_rt::Normalized n;
    kf_resize_rt::Status s = kf_resize_rt::normalize(q, n);
    if (reason) *reason = static_cast<int>(s.reason);
    if (route) *route = s.ok() ? static_cast<int>(n.route.kind) : -1;
    return static_cast<int>(s.code);
}

// The key text a request maps to (default config/target/toolchain, empty source hash),
// written into `out`; returns its length, 0 for the copy route, -1 if unsupported.
extern "C" int kf_rt_key_text(int src_h, int src_w, int dst_w, int dst_h, double fx, double fy, int cn, int is_f32,
                              int interp, char* out, int out_len, unsigned* grid) {
    const std::size_t es = is_f32 ? 4 : 1;
    const kf_resize_ref::Size2 z = [&] {
        try { return kf_resize_ref::resolve_size(src_h, src_w, dst_w, dst_h, fx, fy); }
        catch (...) { return kf_resize_ref::Size2{0, 0, 0, 0}; }
    }();
    if (z.dst_h <= 0 || z.dst_w <= 0) return -1;
    kf_resize_rt::ResizeRequest q;
    const std::size_t sstep = static_cast<std::size_t>(src_w) * cn * es, dstep = static_cast<std::size_t>(z.dst_w) * cn * es;
    q.src = view(0x10000, src_h, src_w, cn, is_f32, sstep, 1);
    q.dst = view(0x10000 + static_cast<std::size_t>(src_h) * sstep + 4096, z.dst_h, z.dst_w, cn, is_f32, dstep, 1);
    q.dsize_w = dst_w; q.dsize_h = dst_h; q.fx = fx; q.fy = fy; q.interpolation = interp;
    kf_resize_rt::Normalized n;
    if (!kf_resize_rt::normalize(q, n).ok()) return -1;
    kf_resize_rt::LaunchShape ls = kf_resize_rt::launch_shape(n, {}, {}, {});
    if (grid) { grid[0] = ls.grid[0]; grid[1] = ls.grid[1]; grid[2] = ls.grid[2]; }
    if (ls.is_copy) return 0;
    const std::string t = ls.key.text();
    if (static_cast<int>(t.size()) + 1 > out_len) return -1;
    std::memcpy(out, t.c_str(), t.size() + 1);
    return static_cast<int>(t.size());
}
