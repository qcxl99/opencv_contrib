// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Runtime JIT, CUDA driver side (P1). See runtime.hpp for the contract.
#include "../precomp.hpp"

#include "runtime.hpp"
#include "compile_protocol.hpp"
#include "cvtcolor.hpp"
#include "reference_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <dlfcn.h>
#include <sys/stat.h>

namespace cv { namespace triton {

namespace {

using namespace kf_resize_rt;
namespace ref = kf_resize_ref;

// ----------------------------------------------------------------- configuration

struct Config
{
    std::string resources, kernels, helper, cache, python, pythonpath, compiler, source_sha256, error;
    // A second kernels source: one file per operator, because a compile key names the file it
    // is compiled from and its sha256, and putting both operators in one file would make
    // every resize edit invalidate every cached cvtColor artifact and vice versa.
    std::string cvt_kernels, cvt_source_sha256, cvt_error;
    // Likewise one file per operator: a warpAffine edit must not invalidate a cached resize
    // or cvtColor artifact.
    std::string wa_kernels, wa_source_sha256, wa_error;
    double timeout_s = 300;
};

std::string envStr(const char* k) { const char* v = std::getenv(k); return v ? std::string(v) : std::string(); }
bool isFile(const std::string& p) { struct stat s; return ::stat(p.c_str(), &s) == 0 && S_ISREG(s.st_mode); }
bool isDir(const std::string& p) { struct stat s; return ::stat(p.c_str(), &s) == 0 && S_ISDIR(s.st_mode); }
void mkdirs(const std::string& p)
{
    for (std::size_t i = 1; i <= p.size(); ++i)
        if (i == p.size() || p[i] == '/')
            ::mkdir(p.substr(0, i).c_str(), 0755);
}
bool hasResources(const std::string& d)
{
    return isFile(d + "/kernels/resize_cpu.py") && isFile(d + "/tools/compile_helper.py");
}

std::string libraryDir()
{
    Dl_info info;
    if (::dladdr(reinterpret_cast<void*>(&runtime::name), &info) && info.dli_fname)
    {
        const std::string p = info.dli_fname;
        const std::size_t s = p.find_last_of('/');
        return s == std::string::npos ? std::string(".") : p.substr(0, s);
    }
    return std::string();
}

const Config& config()
{
    static const Config c = []() {
        Config k;
        // Resources: explicit override, then the install layout relative to this library,
        // then the module source tree (an uninstalled build).
        std::string r = envStr("OPENCV_TRITON_RESOURCES");
        if (r.empty() || !hasResources(r))
        {
            const std::string rel = libraryDir() + "/" + OPENCV_TRITON_REL_RESOURCES;
            r = hasResources(rel) ? rel : std::string(OPENCV_TRITON_SOURCE_RESOURCES);
        }
        k.resources = r;
        k.kernels = r + "/kernels/resize_cpu.py";
        k.helper = r + "/tools/compile_helper.py";
        if (!hasResources(r))
            k.error = "kernels/resize_cpu.py and tools/compile_helper.py not found under " + r;
        // cvtColor is optional: a tree that predates it still runs resize, and asking for a
        // cvtColor code there reports the missing source rather than failing the whole module.
        k.cvt_kernels = r + "/kernels/cvtcolor_cpu.py";
        if (!isFile(k.cvt_kernels))
            k.cvt_error = k.cvt_kernels + " not found";
        k.wa_kernels = r + "/kernels/warpaffine_cpu.py";
        if (!isFile(k.wa_kernels))
            k.wa_error = k.wa_kernels + " not found";
        k.python = envStr("OPENCV_TRITON_PYTHON");
        if (k.python.empty()) k.python = OPENCV_TRITON_PYTHON_DEFAULT;
        k.pythonpath = envStr("OPENCV_TRITON_PYTHONPATH");
        if (k.pythonpath.empty()) k.pythonpath = OPENCV_TRITON_PYTHONPATH_DEFAULT;
        k.compiler = envStr("OPENCV_TRITON_COMPILER");
        if (k.compiler.empty()) k.compiler = OPENCV_TRITON_COMPILER_DEFAULT;
        k.cache = envStr("OPENCV_TRITON_CACHE_DIR");
        if (k.cache.empty())
        {
            const std::string xdg = envStr("XDG_CACHE_HOME"), home = envStr("HOME");
            k.cache = (!xdg.empty() ? xdg : (home.empty() ? std::string("/tmp") : home + "/.cache")) + "/opencv/triton";
        }
        mkdirs(k.cache);
        const std::string t = envStr("OPENCV_TRITON_COMPILE_TIMEOUT");
        if (!t.empty()) k.timeout_s = std::atof(t.c_str());
        if (k.error.empty() && !sha256_file(k.kernels, k.source_sha256))
            k.error = "cannot read " + k.kernels;
        if (k.cvt_error.empty() && !sha256_file(k.cvt_kernels, k.cvt_source_sha256))
            k.cvt_error = "cannot read " + k.cvt_kernels;
        if (k.wa_error.empty() && !sha256_file(k.wa_kernels, k.wa_source_sha256))
            k.wa_error = "cannot read " + k.wa_kernels;
        return k;
    }();
    return c;
}

HelperCommand helperCommand(const Config& c)
{
    HelperCommand h;
    h.python = c.python;
    h.helper = c.helper;
    h.timeout_s = c.timeout_s;
    const std::string path = envStr("PATH"), home = envStr("HOME");
    h.env = {"PATH=" + (path.empty() ? std::string("/usr/bin:/bin") : path), "HOME=" + (home.empty() ? std::string("/tmp") : home),
             "PYTHONPATH=" + c.pythonpath, "CUDA_VISIBLE_DEVICES=", "LC_ALL=C"};   // the helper needs no device
    return h;
}

// --------------------------------------------------------------------- CUDA

std::string cuErr(CUresult r)
{
    const char* s = nullptr;
    cuGetErrorName(r, &s);
    return s ? std::string(s) : ("CUresult " + std::to_string(static_cast<int>(r)));
}

Status error(Reason r, const std::string& msg)
{
    Status s;
    s.code = Code::Error;
    s.reason = r;
    s.message = msg;
    return s;
}

struct CtxGuard
{
    bool pushed = false;
    explicit CtxGuard(CUcontext c)
    {
        CUcontext cur = nullptr;
        cuCtxGetCurrent(&cur);
        if (c && cur != c) pushed = cuCtxPushCurrent(c) == CUDA_SUCCESS;
    }
    ~CtxGuard() { if (pushed) { CUcontext p; cuCtxPopCurrent(&p); } }
};

// Is `c` still the context it was (`gen`, its cuCtxGetId)? A destroyed context took its modules,
// memory and events with it, and its handle value may have come back as another context.
bool contextAlive(CUcontext c, std::uint64_t gen)
{
    unsigned long long id = 0;
    return c && cuCtxGetId(c, &id) == CUDA_SUCCESS && id == gen;
}

bool driverReady()
{
    static const bool ok = cuInit(0) == CUDA_SUCCESS;
    return ok;
}

// --------------------------------------------------------------- resources

std::atomic<long long> g_module_frees{0}, g_table_frees{0};

struct ModuleRes
{
    CUcontext ctx = nullptr;
    std::uint64_t gen = 0;
    CUmodule mod = nullptr;
    CUfunction fn = nullptr;
    Manifest man;
    std::string dir;
    ~ModuleRes()
    {
        if (mod && contextAlive(ctx, gen)) { CtxGuard g(ctx); cuModuleUnload(mod); }
        ++g_module_frees;
    }
};

struct TableRes
{
    CUcontext ctx = nullptr;
    std::uint64_t gen = 0;
    CUdeviceptr base = 0;
    bool stream_ordered = false;          // cuMemAllocAsync memory: freed without draining the context
    std::size_t bytes = 0;
    std::vector<unsigned char> host;      // kept until the resource dies: the async upload reads it
    CUevent uploaded = nullptr;
    CUstream upload_stream = nullptr;
    std::map<std::string, std::size_t> offset;
    int xmax = 0, kx = 0, ky = 0;
    ~TableRes()
    {
        if (contextAlive(ctx, gen))
        {
            CtxGuard g(ctx);
            // Normally every launch that read these tables has completed. Only a table whose
            // launch failed can still have its upload pending: keep allocate -> upload -> free
            // in order rather than free under it.
            if (uploaded && cuEventQuery(uploaded) == CUDA_ERROR_NOT_READY)
                cuStreamWaitEvent(nullptr, uploaded, 0);
            if (uploaded) cuEventDestroy(uploaded);
            // cuMemFree blocks until the context's in-flight work drains (measured), which would
            // stall the application's thread on every eviction; the stream-ordered free does not.
            if (base) { if (stream_ordered) cuMemFreeAsync(base, nullptr); else cuMemFree(base); }
        }
        ++g_table_frees;
    }
    CUdeviceptr at(const char* k) const { return base + offset.at(k); }
};

struct Pending
{
    CUcontext ctx;
    std::uint64_t gen;
    CUevent done;
    std::vector<std::shared_ptr<void>> keep;   // what this launch reads: freed only after `done`
};

template <class V>
struct Lru
{
    std::size_t cap;
    std::list<std::pair<std::string, std::shared_ptr<V>>> items;
    std::unordered_map<std::string, typename std::list<std::pair<std::string, std::shared_ptr<V>>>::iterator> index;
    explicit Lru(std::size_t c) : cap(c) {}
    std::shared_ptr<V> get(const std::string& k)
    {
        auto it = index.find(k);
        if (it == index.end()) return nullptr;
        items.splice(items.begin(), items, it->second);
        return it->second->second;
    }
    void put(const std::string& k, std::shared_ptr<V> v, std::vector<std::shared_ptr<V>>& evicted)
    {
        items.emplace_front(k, std::move(v));
        index[k] = items.begin();
        shrink(evicted);
    }
    void shrink(std::vector<std::shared_ptr<V>>& evicted)
    {
        while (items.size() > cap)
        {
            evicted.push_back(std::move(items.back().second));
            index.erase(items.back().first);
            items.pop_back();
        }
    }
};

// Resources a captured graph depends on. A graph outlives the call that recorded it, so the
// module binary and the table buffers must outlive it too -- and the caches cannot guarantee
// that, since an eviction may happen before the graph is ever launched.
struct CapturePin
{
    CUcontext ctx = nullptr;
    std::uint64_t gen = 0;
    std::vector<std::shared_ptr<void>> keep;
};

struct State
{
    std::mutex mu;
    Lru<ModuleRes> modules{64};
    Lru<TableRes> tables{256};
    std::list<Pending> pending;
    std::vector<CapturePin> capture_pins;   // held until releaseCapturePins(), see there
    detail::RuntimeStats st;
    std::map<std::string, std::shared_ptr<std::mutex>> compile_mu;
    bool poison = false;
};

State& S()
{
    static State* s = new State();   // never destroyed: process exit may follow CUDA teardown
    return *s;
}

// Counts only. The caller's vector is destroyed AFTER it releases the state mutex: unloading a
// module blocks until the context's in-flight work has drained (measured), and must not hold
// up every other thread using the runtime.
template <class V>
void countEvicted(const std::vector<std::shared_ptr<V>>& ev, long long& counter, long long& deferred)
{
    for (const auto& r : ev)
    {
        ++counter;
        if (r.use_count() > 1) ++deferred;   // an in-flight launch still holds it
    }
}

// True when `stream` is inside a capture region. cuStreamIsCapturing is legal there, unlike
// cuEventQuery: querying any event while capturing invalidates the whole capture
// (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, measured -- runs/cvtcolor-module-0001/bench/
// probe_capture_min2.cpp).
bool capturing(CUstream stream)
{
    if (!stream) return false;
    CUstreamCaptureStatus st = CU_STREAM_CAPTURE_STATUS_NONE;
    return cuStreamIsCapturing(stream, &st) == CUDA_SUCCESS && st != CU_STREAM_CAPTURE_STATUS_NONE;
}

// Finish a launch: publish the counters and keep whatever the launch reads alive.
//
// Two regimes, and they must not be confused:
//
//  * normal -- a completion event is recorded and the module/tables move to `pending`; they are
//    freed once that event signals (reapLocked).
//
//  * capturing -- the work is not executing, it is being recorded, so there is no completion to
//    wait for. Creating an event here would also put a capture-scoped event into `pending`, and
//    the NEXT capture that waits on it fails with CUDA_ERROR_STREAM_CAPTURE_ISOLATION
//    (measured: runs/cvtcolor-module-0001/bench/probe_graph2.cpp). But the graph produced by
//    this capture may be launched much later, after the LRU has evicted everything it read, so
//    the module and tables are PINNED instead of being left to the cache. Without this, a
//    replay after an eviction would reference an unloaded module or a freed table buffer.
//
// Pins are released only by releaseCapturePins(). CUDA gives no notification when the caller
// destroys a graph, so the module cannot know; a caller that keeps graphs indefinitely keeps
// the pin, which is bounded by the number of distinct (code, size-class) keys it captured and is
// a few kilobytes each.
// `tab2` is a SECOND table set for the same launch: cvtColor's Xyz/Lab/Luv codes read the small
// sdiv/hdiv set and their own 2.3 MB family set in the same kernel, and both must be kept alive
// exactly as long as `tab`. Null for every other caller.
Status finishLaunch(State& s, CUcontext ctx, std::uint64_t gen, CUstream stream,
                    const std::shared_ptr<ModuleRes>& mod, const std::shared_ptr<TableRes>& tab,
                    const std::string& route, const std::string& key,
                    const std::string& artifact,
                    const std::shared_ptr<TableRes>& tab2 = nullptr)
{
    std::lock_guard<std::mutex> lk(s.mu);
    ++s.st.launches;
    s.st.last_route = route;
    s.st.last_key = key;
    s.st.last_artifact = artifact;

    if (capturing(stream))
    {
        CapturePin* pin = nullptr;
        for (CapturePin& p : s.capture_pins)
            if (p.ctx == ctx && p.gen == gen) { pin = &p; break; }
        if (!pin)
        {
            s.capture_pins.push_back(CapturePin{ctx, gen, {}});
            pin = &s.capture_pins.back();
        }
        auto add = [&](const std::shared_ptr<void>& r) {
            if (!r) return;
            for (const auto& have : pin->keep)
                if (have.get() == r.get()) return;
            pin->keep.push_back(r);
        };
        add(std::static_pointer_cast<void>(mod));
        add(std::static_pointer_cast<void>(tab));
        add(std::static_pointer_cast<void>(tab2));
        s.st.capture_pin_holds = static_cast<long long>(pin->keep.size());
        s.st.capture_launches++;
        Status ok;
        ok.code = Code::Submitted;   // Status{} is Code::Unsupported, not "success"
        return ok;
    }

    CUevent done = nullptr;
    CUresult r = cuEventCreate(&done, CU_EVENT_DISABLE_TIMING);
    if (r == CUDA_SUCCESS) r = cuEventRecord(done, stream);
    if (r != CUDA_SUCCESS)
    {
        // Cannot observe completion: keep the resources for the life of the process rather than
        // risk freeing what the launch may still read.
        static std::vector<std::shared_ptr<void>> leaked;
        leaked.push_back(mod);
        if (tab) leaked.push_back(tab);
        if (tab2) leaked.push_back(tab2);
        Status e = error(Reason::LaunchFailed, "completion event: " + cuErr(r));
        e.submitted = true;
        return e;
    }
    Pending p;
    p.ctx = ctx;
    p.gen = gen;
    p.done = done;
    p.keep.push_back(mod);
    if (tab) p.keep.push_back(tab);
    if (tab2) p.keep.push_back(tab2);
    s.pending.push_back(std::move(p));
    s.st.pending = static_cast<long long>(s.pending.size());
    Status ok;
    ok.code = Code::Submitted;
    return ok;
}

// Same rule: what a completed launch held is moved to `released`, destroyed after the unlock.
//
// `skipQuery` is set when the caller is capturing: reaping is a bookkeeping optimisation, and
// doing it must never be allowed to break a capture. Pending entries simply stay pending until
// a later, non-capturing call.
void reapLocked(State& s, std::vector<std::shared_ptr<void>>& released, bool skipQuery = false)
{
    for (auto it = s.pending.begin(); it != s.pending.end();)
    {
        if (contextAlive(it->ctx, it->gen))
        {
            CtxGuard g(it->ctx);
            if (skipQuery || cuEventQuery(it->done) == CUDA_ERROR_NOT_READY) { ++it; continue; }
            cuEventDestroy(it->done);
        }
        released.insert(released.end(), it->keep.begin(), it->keep.end());
        it = s.pending.erase(it);             // complete, or its context is gone: release
    }
    s.st.pending = static_cast<long long>(s.pending.size());
}

Status currentContext(int device, CUcontext& ctx, std::uint64_t& gen)
{
    if (!driverReady()) return error(Reason::ToolchainUnavailable, "cuInit failed");
    ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    if (!ctx)
    {
        CUdevice d;
        CUresult r = cuDeviceGet(&d, device);
        if (r == CUDA_SUCCESS) r = cuDevicePrimaryCtxRetain(&ctx, d);
        if (r == CUDA_SUCCESS) r = cuCtxSetCurrent(ctx);
        if (r != CUDA_SUCCESS) return error(Reason::ContextMismatch, "no context for device " + std::to_string(device) + ": " + cuErr(r));
    }
    CUdevice cd;
    if (cuCtxGetDevice(&cd) != CUDA_SUCCESS || static_cast<int>(cd) != device)
        return error(Reason::ContextMismatch, "the current context is not on device " + std::to_string(device));
    unsigned long long id = 0;
    if (cuCtxGetId(ctx, &id) != CUDA_SUCCESS) return error(Reason::ContextMismatch, "cuCtxGetId failed");
    gen = id;   // unique for the context's lifetime and never reused, unlike the handle value
    Status ok;
    ok.code = Code::Prepared;
    return ok;
}

// ---------------------------------------------------------------- modules

std::shared_ptr<ModuleRes> moduleFor(const KernelKey& key, CUcontext ctx, std::uint64_t gen, Status& st,
                                const std::string& kernels_source)
{
    State& s = S();
    const std::string text = key.text();
    const std::string mk = std::to_string(gen) + "|" + text;
    std::shared_ptr<std::mutex> kmu;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        if (auto m = s.modules.get(mk)) { ++s.st.module_hits; return m; }
        auto& slot = s.compile_mu[text];
        if (!slot) slot = std::make_shared<std::mutex>();
        kmu = slot;
    }
    std::lock_guard<std::mutex> klk(*kmu);   // one lookup / compile per key in this process
    {
        std::lock_guard<std::mutex> lk(s.mu);
        if (auto m = s.modules.get(mk)) { ++s.st.module_hits; return m; }
    }
    const Config& c = config();
    const std::string dir = c.cache + "/" + key_digest(key);
    Manifest man;
    std::string why;
    bool from_disk = verify_artifact(dir, text, man, why);
    if (!from_disk)
    {
        if (isDir(dir)) { std::lock_guard<std::mutex> lk(s.mu); ++s.st.disk_rejected; }
        CompileOutcome o = compile_key(helperCommand(c), key, kernels_source, c.cache);
        if (!o.status.ok()) { st = o.status; return nullptr; }
        {
            std::lock_guard<std::mutex> lk(s.mu);
            if (o.kind == "compiled") ++s.st.helper_compiles; else ++s.st.helper_cached;
        }
        if (!verify_artifact(o.artifact_dir, text, man, why)) { st = error(Reason::ManifestInvalid, why); return nullptr; }
    }
    if (!launch_abi_ok(man, why)) { st = error(Reason::ManifestInvalid, why); return nullptr; }   // before any launch
    std::string bin;
    if (!read_file(dir + "/" + man.binary_file, bin) || sha256_hex(bin) != man.binary_sha256)
    {
        st = error(Reason::ManifestInvalid, "binary changed after verification");
        return nullptr;
    }
    auto m = std::make_shared<ModuleRes>();
    m->ctx = ctx;
    m->gen = gen;
    m->man = man;
    m->dir = dir;
    CUresult r = cuModuleLoadData(&m->mod, bin.data());
    if (r == CUDA_SUCCESS) r = cuModuleGetFunction(&m->fn, m->mod, man.function.c_str());
    if (r == CUDA_SUCCESS && man.shared_bytes > 48 * 1024)
        r = cuFuncSetAttribute(m->fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, man.shared_bytes);
    if (r != CUDA_SUCCESS) { st = error(Reason::LoadFailed, "loading " + dir + ": " + cuErr(r)); return nullptr; }
    std::vector<std::shared_ptr<ModuleRes>> ev;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        ++s.st.module_loads;
        if (from_disk) ++s.st.disk_hits;
        s.modules.put(mk, m, ev);
        countEvicted(ev, s.st.module_evictions, s.st.deferred);
    }
    return m;
}

// ----------------------------------------------------------------- tables

void pack(std::vector<unsigned char>& host, std::map<std::string, std::size_t>& off, const char* name,
          const void* data, std::size_t bytes)
{
    const std::size_t at = (host.size() + 255) / 256 * 256;
    host.resize(at + bytes);
    if (bytes) std::memcpy(host.data() + at, data, bytes);
    off[name] = at;
}

// The tables the P1 kernels read, built by the portable oracle's plan half, exactly as
// operators/resize/kernel.py::_cpu_plan builds them.
void buildTables(const Normalized& n, TableRes& t)
{
    const ref::Route k = n.route.kind;
    if (k == ref::ROUTE_NEAREST)
    {
        const double ifx = 1.0 / n.inv_x, ify = 1.0 / n.inv_y;   // resizeNN: 1./fx
        std::vector<int> xo(n.dst_w), yo(n.dst_h);
        for (int x = 0; x < n.dst_w; ++x) xo[x] = std::min(static_cast<int>(std::floor(x * ifx)), n.src_w - 1);
        for (int y = 0; y < n.dst_h; ++y) yo[y] = std::min(static_cast<int>(std::floor(y * ify)), n.src_h - 1);
        pack(t.host, t.offset, "xofs", xo.data(), xo.size() * 4);
        pack(t.host, t.offset, "yofs", yo.data(), yo.size() * 4);
    }
    else if (k == ref::ROUTE_LINEAR || k == ref::ROUTE_CUBIC)
    {
        const int ks = k == ref::ROUTE_CUBIC ? 4 : 2;
        const ref::Axis ax = ref::generic_axis(n.dst_w, n.src_w, n.inv_x, ks, n.route.area_mode, true);
        const ref::Axis ay = ref::generic_axis(n.dst_h, n.src_h, n.inv_y, ks, n.route.area_mode, false);
        t.xmax = ax.xmax;
        pack(t.host, t.offset, "xofs", ax.ofs.data(), ax.ofs.size() * 4);
        pack(t.host, t.offset, "yofs", ay.ofs.data(), ay.ofs.size() * 4);
        if (n.type == ElemType::F32)
        {
            pack(t.host, t.offset, "ax", ax.coeffs.data(), ax.coeffs.size() * 4);
            pack(t.host, t.offset, "by", ay.coeffs.data(), ay.coeffs.size() * 4);
            pack(t.host, t.offset, "byf", ay.coeffs.data(), ay.coeffs.size() * 4);
        }
        else
        {
            const std::vector<int> axi = ref::fixed_coeffs(ax.coeffs), byi = ref::fixed_coeffs(ay.coeffs);
            std::vector<float> byf(byi.size());
            const float sc = 1.0f / static_cast<float>(2048 * 2048);   // exact power of two
            for (std::size_t i = 0; i < byi.size(); ++i) byf[i] = static_cast<float>(byi[i]) * sc;
            pack(t.host, t.offset, "ax", axi.data(), axi.size() * 4);
            pack(t.host, t.offset, "by", byi.data(), byi.size() * 4);
            pack(t.host, t.offset, "byf", byf.data(), byf.size() * 4);
        }
    }
    else if (k == ref::ROUTE_AREA)
    {
        const ref::AreaTab tx = ref::area_tab(n.src_w, n.dst_w, 1.0 / n.inv_x);
        const ref::AreaTab ty = ref::area_tab(n.src_h, n.dst_h, 1.0 / n.inv_y);
        t.kx = tx.taps; t.ky = ty.taps;
        pack(t.host, t.offset, "xs", tx.src_index.data(), tx.src_index.size() * sizeof(int));
        pack(t.host, t.offset, "xa", tx.alpha.data(), tx.alpha.size() * sizeof(float));
        pack(t.host, t.offset, "xc", tx.count.data(), tx.count.size() * sizeof(int));
        pack(t.host, t.offset, "ys", ty.src_index.data(), ty.src_index.size() * sizeof(int));
        pack(t.host, t.offset, "ya", ty.alpha.data(), ty.alpha.size() * sizeof(float));
        pack(t.host, t.offset, "yc", ty.count.data(), ty.count.size() * sizeof(int));
    }
    if (t.host.empty()) t.host.resize(16);
}

std::shared_ptr<TableRes> tablesFor(const Normalized& n, CUcontext ctx, std::uint64_t gen, CUstream stream, Status& st)
{
    State& s = S();
    char key[256];
    std::snprintf(key, sizeof key, "%llu|%d|%d|%d|%d|%d|%a|%a|%d|%d|%d", static_cast<unsigned long long>(gen),
                  static_cast<int>(n.route.kind), n.src_h, n.src_w, n.dst_h, n.dst_w, n.inv_y, n.inv_x,
                  static_cast<int>(n.type), n.route.area_mode ? 1 : 0, n.channels);
    bool poison;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        if (auto t = s.tables.get(key))
        {
            ++s.st.table_hits;
            if (t->upload_stream != stream)
            {
                const CUresult r = cuStreamWaitEvent(stream, t->uploaded, 0);   // upload -> this launch
                if (r != CUDA_SUCCESS) { st = error(Reason::LaunchFailed, "cuStreamWaitEvent: " + cuErr(r)); return nullptr; }
                ++s.st.cross_stream_waits;
            }
            return t;
        }
        poison = s.poison;
    }
    auto t = std::make_shared<TableRes>();
    t->ctx = ctx;
    t->gen = gen;
    buildTables(n, *t);
    t->bytes = t->host.size();
    CUdevice dev;
    int pools = 0;
    if (cuCtxGetDevice(&dev) == CUDA_SUCCESS)
        cuDeviceGetAttribute(&pools, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, dev);
    t->stream_ordered = pools != 0;
    CUresult r = t->stream_ordered ? cuMemAllocAsync(&t->base, t->bytes, stream) : cuMemAlloc(&t->base, t->bytes);
    if (r == CUDA_SUCCESS && poison) r = cuMemsetD8Async(t->base, 0xFF, t->bytes, stream);
    if (r == CUDA_SUCCESS) r = cuMemcpyHtoDAsync(t->base, t->host.data(), t->bytes, stream);
    if (r == CUDA_SUCCESS) r = cuEventCreate(&t->uploaded, CU_EVENT_DISABLE_TIMING);
    if (r == CUDA_SUCCESS) r = cuEventRecord(t->uploaded, stream);
    if (r != CUDA_SUCCESS) { st = error(Reason::DeviceAlloc, "table upload: " + cuErr(r)); return nullptr; }
    t->upload_stream = stream;
    std::vector<std::shared_ptr<TableRes>> ev;
    std::lock_guard<std::mutex> lk(s.mu);
    ++s.st.table_uploads;
    s.tables.put(key, t, ev);
    countEvicted(ev, s.st.table_evictions, s.st.deferred);
    return t;
}

// ----------------------------------------------------------------- launch

struct Arg
{
    char kind = 'p';   // p pointer, i int32, f float32
    CUdeviceptr p = 0;
    int i = 0;
    float f = 0;
};

Status launch(const ModuleRes& m, const std::map<std::string, Arg>& args, const unsigned grid[3], CUstream stream)
{
    std::vector<Arg> values(m.man.abi.size());
    std::vector<void*> params(m.man.abi.size());
    for (std::size_t i = 0; i < m.man.abi.size(); ++i)
    {
        const ManifestParam& p = m.man.abi[i];
        Arg& v = values[i];
        if (p.null_pointer) { v.kind = 'p'; v.p = 0; }   // allowed by launch_abi_ok() at load
        else
        {
            auto it = args.find(p.name);
            if (it == args.end()) return error(Reason::ManifestInvalid, "no value for kernel parameter " + p.name);
            v = it->second;
            const char want = p.type[0] == '*' ? 'p' : (p.type == "i32" ? 'i' : (p.type == "fp32" ? 'f' : '?'));
            if (want != v.kind) return error(Reason::ManifestInvalid, "parameter " + p.name + " has type " + p.type);
        }
        params[i] = v.kind == 'p' ? static_cast<void*>(&v.p) : (v.kind == 'i' ? static_cast<void*>(&v.i) : static_cast<void*>(&v.f));
    }
    const CUresult r = cuLaunchKernel(m.fn, grid[0], grid[1], grid[2], static_cast<unsigned>(m.man.threads_per_block), 1, 1,
                                      static_cast<unsigned>(m.man.shared_bytes), stream, params.data(), nullptr);
    if (r != CUDA_SUCCESS) return error(Reason::LaunchFailed, "cuLaunchKernel: " + cuErr(r));
    Status ok;
    ok.code = Code::Submitted;
    ok.submitted = true;
    return ok;
}

// The div tables RGB2HSV_b reads. Keyed by the code, because the hue range (180 or 256)
// changes hdiv while sdiv stays the same; two codes with the same range share one upload.
std::shared_ptr<TableRes> cvtTablesFor(const kf_cvtcolor::HsvTables& hv, int code, CUcontext ctx,
                                       std::uint64_t gen, CUstream stream, Status& st)
{
    State& s = S();
    const std::string key = std::to_string(gen) + "|cvt|" + std::to_string(code);
    bool poison;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        if (auto t = s.tables.get(key))
        {
            ++s.st.table_hits;
            if (t->upload_stream != stream)
            {
                const CUresult r = cuStreamWaitEvent(stream, t->uploaded, 0);
                if (r != CUDA_SUCCESS) { st = error(Reason::LaunchFailed, "cuStreamWaitEvent: " + cuErr(r)); return nullptr; }
                ++s.st.cross_stream_waits;
            }
            return t;
        }
        poison = s.poison;
    }
    auto t = std::make_shared<TableRes>();
    t->ctx = ctx;
    t->gen = gen;
    pack(t->host, t->offset, "sdiv", hv.sdiv.data(), hv.sdiv.size() * sizeof(int));
    pack(t->host, t->offset, "hdiv", hv.hdiv.data(), hv.hdiv.size() * sizeof(int));
    // The Xyz/Lab/Luv kernels take three more table pointers whether or not their op reads them,
    // so every row needs SOME live address for each: a dead branch that Triton folds away would
    // never dereference it, but passing 0 would also be wrong the first time a fold did not
    // happen. Eight bytes of a shared static buffer covers all three.
    static int dummy = 0;
    pack(t->host, t->offset, "t32", &dummy, sizeof(int));
    pack(t->host, t->offset, "t16", &dummy, sizeof(int));
    pack(t->host, t->offset, "t64", &dummy, sizeof(int));
    t->bytes = t->host.size();
    CUdevice dev;
    int pools = 0;
    if (cuCtxGetDevice(&dev) == CUDA_SUCCESS)
        cuDeviceGetAttribute(&pools, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, dev);
    t->stream_ordered = pools != 0;
    CUresult r = t->stream_ordered ? cuMemAllocAsync(&t->base, t->bytes, stream) : cuMemAlloc(&t->base, t->bytes);
    if (r == CUDA_SUCCESS && poison) r = cuMemsetD8Async(t->base, 0xFF, t->bytes, stream);
    if (r == CUDA_SUCCESS) r = cuMemcpyHtoDAsync(t->base, t->host.data(), t->bytes, stream);
    if (r == CUDA_SUCCESS) r = cuEventCreate(&t->uploaded, CU_EVENT_DISABLE_TIMING);
    if (r == CUDA_SUCCESS) r = cuEventRecord(t->uploaded, stream);
    if (r != CUDA_SUCCESS) { st = error(Reason::DeviceAlloc, "cvtColor table upload: " + cuErr(r)); return nullptr; }
    t->upload_stream = stream;
    std::vector<std::shared_ptr<TableRes>> ev;
    std::lock_guard<std::mutex> lk(s.mu);
    ++s.st.table_uploads;
    s.tables.put(key, t, ev);
    countEvicted(ev, s.st.table_evictions, s.st.deferred);
    return t;
}

bool p1Route(ref::Route r)
{
    return r == ref::ROUTE_COPY || r == ref::ROUTE_NEAREST || r == ref::ROUTE_LINEAR || r == ref::ROUTE_CUBIC ||
           r == ref::ROUTE_AREA_FAST || r == ref::ROUTE_AREA;
}

// The Xyz/Lab/Luv tables. They are ~2.3 MB in total (the 33^3 Luv grid alone is 1.7 MB), so they
// get their own upload keyed on the FAMILY rather than the code: BGR2Lab and RGB2Lab read the
// same bytes, and so do BGR2XYZ/RGB2XYZ and the four reverse codes. Keying by code would upload
// the same 1.7 MB twice for BGR2Luv and RGB2Luv.
//
// The packing order is the contract with the kernel (kernels/cvtcolor_cpu.py: T_GAMMA, T_CBRT,
// ...) and with kf_cvtcolor::LabLuvLayout. Both sides are checked by the numeric tests; a
// mismatch shows up as every pixel being wrong, not as a subtle drift.
std::shared_ptr<TableRes> labluvTablesFor(int code, CUcontext ctx, std::uint64_t gen,
                                          CUstream stream, Status& st)
{
    // family: 0 = XYZ (no tables), 1 = Lab/Luv reverse + Lab forward (the small tables),
    // 2 = Luv forward (the big grid). 2 implies 1, so Luv forward gets both.
    const kf_cvtcolor::CodeSpec* spec = kf_cvtcolor::find(code);
    // Only Op::Lab and Op::Luv read these tables. Every other op -- and XYZ, whose coefficients
    // are constexprs -- gets NO table set at all, which matters: this cache is shared with the
    // per-code sdiv/hdiv sets and is small (256), so uploading a 2.3 MB set for a code that never
    // reads it would evict live entries and turn a table hit into a load on every call.
    if (!spec || (spec->op != kf_cvtcolor::Op::Lab && spec->op != kf_cvtcolor::Op::Luv))
        return nullptr;
    const bool isLuvFwd = spec->op == kf_cvtcolor::Op::Luv && !spec->inverse;
    const std::string key = std::to_string(gen) + "|labluv|" + (isLuvFwd ? "2" : "1");

    State& s = S();
    bool poison;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        if (auto t = s.tables.get(key))
        {
            ++s.st.table_hits;
            if (t->upload_stream != stream)
            {
                const CUresult r = cuStreamWaitEvent(stream, t->uploaded, 0);
                if (r != CUDA_SUCCESS) { st = error(Reason::LaunchFailed, "cuStreamWaitEvent: " + cuErr(r)); return nullptr; }
                ++s.st.cross_stream_waits;
            }
            return t;
        }
        poison = s.poison;
    }

    const kf_cvtcolor::LabLuvTables& tab = kf_cvtcolor::labluv_tables();
    auto t = std::make_shared<TableRes>();
    t->ctx = ctx;
    t->gen = gen;
    // t32: fixed-layout concat, in the order the kernel's offsets name. `pack` aligns each blob
    // to 256 bytes, which happens to match these offsets (they are all multiples of 256 entries
    // x 4 bytes) -- if an offset is ever added that is not, this needs a manual layout instead.
    {
        std::vector<int> t32(static_cast<std::size_t>(kf_cvtcolor::LabLuvLayout::t32_entries), 0);
        using L = kf_cvtcolor::LabLuvLayout;
        std::copy(tab.gamma.begin(), tab.gamma.end(), t32.begin() + L::gamma);
        std::copy(tab.cbrt.begin(), tab.cbrt.end(), t32.begin() + L::cbrt);
        std::copy(tab.yf.begin(), tab.yf.end(), t32.begin() + L::yf);
        std::copy(tab.abxz.begin(), tab.abxz.end(), t32.begin() + L::abxz);
        std::copy(tab.invgamma.begin(), tab.invgamma.end(), t32.begin() + L::invgamma);
        std::copy(tab.lu.begin(), tab.lu.end(), t32.begin() + L::lu);
        std::copy(tab.lv.begin(), tab.lv.end(), t32.begin() + L::lv);
        pack(t->host, t->offset, "t32", t32.data(), t32.size() * sizeof(int));
    }
    pack(t->host, t->offset, "t64", tab.lvpl.data(), tab.lvpl.size() * sizeof(long long));
    if (isLuvFwd)
    {
        pack(t->host, t->offset, "t16", tab.luvLut.data(), tab.luvLut.size() * sizeof(short));
    }
    else
    {
        // Luv's LUT is not read by any other code; keep the pointer live with one element so the
        // argument is never null.
        static const short one = 0;
        pack(t->host, t->offset, "t16", &one, sizeof(short));
    }
    t->bytes = t->host.size();

    CUdevice dev;
    int pools = 0;
    if (cuCtxGetDevice(&dev) == CUDA_SUCCESS)
        cuDeviceGetAttribute(&pools, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, dev);
    t->stream_ordered = pools != 0;
    CUresult r = t->stream_ordered ? cuMemAllocAsync(&t->base, t->bytes, stream) : cuMemAlloc(&t->base, t->bytes);
    if (r == CUDA_SUCCESS && poison) r = cuMemsetD8Async(t->base, 0xFF, t->bytes, stream);
    if (r == CUDA_SUCCESS) r = cuMemcpyHtoDAsync(t->base, t->host.data(), t->bytes, stream);
    if (r == CUDA_SUCCESS) r = cuEventCreate(&t->uploaded, CU_EVENT_DISABLE_TIMING);
    if (r == CUDA_SUCCESS) r = cuEventRecord(t->uploaded, stream);
    if (r != CUDA_SUCCESS) { st = error(Reason::DeviceAlloc, "cvtColor Lab/Luv table upload: " + cuErr(r)); return nullptr; }
    t->upload_stream = stream;
    std::vector<std::shared_ptr<TableRes>> ev;
    std::lock_guard<std::mutex> lk(s.mu);
    ++s.st.table_uploads;
    s.tables.put(key, t, ev);
    countEvicted(ev, s.st.table_evictions, s.st.deferred);
    return t;
}

}  // namespace

namespace runtime {

const char* name() { return "cuda-jit"; }

bool deviceAvailable(int device)
{
    if (!driverReady()) return false;
    int n = 0;
    return cuDeviceGetCount(&n) == CUDA_SUCCESS && device >= 0 && device < n;
}

Status resize(const ResizeRequest& req, const ExecContext& ex, std::string* routeName)
{
    Normalized n;
    Status st = normalize(req, n);
    if (!st.ok()) return st;
    if (routeName) *routeName = ref::route_name(n.route.kind);
    if (!p1Route(n.route.kind))
    {
        Status u;
        u.code = Code::Unsupported;
        u.reason = Reason::RouteNotEnabled;
        u.message = std::string("route ") + ref::route_name(n.route.kind) + " is not served by this build (P1: copy, nearest, linear, area_fast)";
        return u;
    }
    const Config& c = config();
    if (!c.error.empty()) return error(Reason::ToolchainUnavailable, c.error);
    CUcontext ctx;
    std::uint64_t gen = 0;
    st = currentContext(ex.ctx.device, ctx, gen);
    if (!st.ok()) return st;
    const CUstream stream = reinterpret_cast<CUstream>(ex.stream);
    State& s = S();
    std::vector<std::shared_ptr<void>> released;   // destroyed at return, outside the lock
    {
        std::lock_guard<std::mutex> lk(s.mu);
        reapLocked(s, released, capturing(stream));
    }
    if (n.route.kind == ref::ROUTE_COPY)   // cv::resize: src.copyTo(dst). A copy, not a Triton launch.
    {
        CUDA_MEMCPY2D cp;
        std::memset(&cp, 0, sizeof cp);
        cp.srcMemoryType = CU_MEMORYTYPE_DEVICE; cp.srcDevice = req.src.ptr; cp.srcPitch = req.src.step;
        cp.dstMemoryType = CU_MEMORYTYPE_DEVICE; cp.dstDevice = req.dst.ptr; cp.dstPitch = req.dst.step;
        cp.WidthInBytes = req.src.row_bytes(); cp.Height = static_cast<std::size_t>(req.src.rows);
        const CUresult r = cuMemcpy2DAsync(&cp, stream);
        if (r != CUDA_SUCCESS) return error(Reason::LaunchFailed, "cuMemcpy2DAsync: " + cuErr(r));
        std::lock_guard<std::mutex> lk(s.mu);
        ++s.st.copies;
        s.st.last_route = "copy";
        Status ok;
        ok.code = Code::Submitted;
        ok.submitted = true;
        return ok;
    }

    CUdevice dev;
    cuCtxGetDevice(&dev);
    int major = 0, minor = 0, warp = 32;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);
    Target target;
    target.backend = "cuda";
    target.arch = major * 10 + minor;
    target.warp_size = warp;
    Toolchain tc;
    tc.compiler = c.compiler;
    tc.source_sha256 = c.source_sha256;
    const LaunchShape shape = launch_shape(n, LaunchConfig{}, target, tc);

    std::shared_ptr<ModuleRes> mod = moduleFor(shape.key, ctx, gen, st, c.kernels);
    if (!mod) return st;
    std::shared_ptr<TableRes> tab;
    if (n.route.kind != ref::ROUTE_AREA_FAST)
    {
        tab = tablesFor(n, ctx, gen, stream, st);
        if (!tab) return st;
    }

    const std::size_t es = elem_size(n.type);
    std::map<std::string, Arg> a;
    auto ptr = [&](const char* k, CUdeviceptr v) { Arg x; x.kind = 'p'; x.p = v; a[k] = x; };
    auto i32 = [&](const char* k, long long v) { Arg x; x.kind = 'i'; x.i = static_cast<int>(v); a[k] = x; };
    auto f32 = [&](const char* k, float v) { Arg x; x.kind = 'f'; x.f = v; a[k] = x; };
    ptr("src", req.src.ptr);
    ptr("dst", req.dst.ptr);
    i32("src_h", n.src_h); i32("src_w", n.src_w); i32("dst_h", n.dst_h);
    i32("W", static_cast<long long>(n.dst_w) * n.channels);
    i32("src_rs", static_cast<long long>(req.src.step / es)); i32("dst_rs", static_cast<long long>(req.dst.step / es));
    i32("src_bs", 0); i32("dst_bs", 0);
    if (n.route.kind == ref::ROUTE_NEAREST)
    {
        ptr("xofs", tab->at("xofs")); ptr("yofs", tab->at("yofs"));
    }
    else if (n.route.kind == ref::ROUTE_LINEAR || n.route.kind == ref::ROUTE_CUBIC)
    {
        ptr("xofs", tab->at("xofs")); ptr("ax", tab->at("ax")); ptr("yofs", tab->at("yofs"));
        ptr("by", tab->at("by")); ptr("byf", tab->at("byf")); i32("xmax", tab->xmax);
    }
    else if (n.route.kind == ref::ROUTE_AREA)
    {
        ptr("xs", tab->at("xs")); ptr("xa", tab->at("xa")); ptr("xc", tab->at("xc"));
        ptr("ys", tab->at("ys")); ptr("ya", tab->at("ya")); ptr("yc", tab->at("yc"));
        i32("KX", tab->kx); i32("KY", tab->ky);
    }
    else  // area_fast
    {
        const int ix = n.route.ix, iy = n.route.iy;
        i32("dwidth1", static_cast<long long>(n.src_w / ix) * n.channels);
        f32("scale", 1.0f / static_cast<float>(ix * iy));
        i32("IX", ix); i32("IY", iy);
    }

    st = launch(*mod, a, shape.grid, stream);
    if (!st.ok()) return st;
    return finishLaunch(s, ctx, gen, stream, mod, tab, ref::route_name(n.route.kind),
                        shape.key.text(), mod->dir);
}


// ------------------------------------------------------------------- cvtColor

// One flat pass over output pixels. Everything that decides what is computed lives in
// cvtcolor.cpp (the code table, the validation, the compile key); this only publishes the
// two div tables and launches.
Status cvtColor(const kf_cvtcolor::Request& req, const ExecContext& ex, std::string* opName)
{
    kf_cvtcolor::Normalized n;
    Status st = kf_cvtcolor::normalize(req, n);
    if (!st.ok()) return st;
    const kf_cvtcolor::CodeSpec& spec = *n.spec;
    if (opName) *opName = spec.name;

    const Config& c = config();
    if (!c.error.empty()) return error(Reason::ToolchainUnavailable, c.error);
    if (!c.cvt_error.empty()) return error(Reason::ToolchainUnavailable, c.cvt_error);
    CUcontext ctx;
    std::uint64_t gen = 0;
    st = currentContext(ex.ctx.device, ctx, gen);
    if (!st.ok()) return st;
    const CUstream stream = reinterpret_cast<CUstream>(ex.stream);
    State& s = S();
    std::vector<std::shared_ptr<void>> released;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        reapLocked(s, released, capturing(stream));
    }

    CUdevice dev;
    cuCtxGetDevice(&dev);
    int major = 0, minor = 0, warp = 32;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);

    // The key, built exactly as cvtcolor.cpp describes it. The kernel's constexpr list is
    // spelled out here because KernelKey stores it as name/value pairs.
    KernelKey key;
    key.kernel = "_cvtcolor_cpu_kernel";
    key.signature = kf_cvtcolor::signature(n);
    key.cfg.block_h = 1;
    key.cfg.block_w = kf_cvtcolor::block_size(n);
    key.cfg.num_warps = kf_cvtcolor::num_warps(n);
    key.cfg.num_stages = 3;
    key.cfg.ry = 1;
    key.target.backend = "cuda";
    key.target.arch = major * 10 + minor;
    key.target.warp_size = warp;
    key.toolchain.compiler = c.compiler;
    key.toolchain.source_sha256 = c.cvt_source_sha256;
    key.constexprs = {{"BLOCK", key.cfg.block_w},
                      {"BLUE", spec.blue},
                      {"DCN", spec.dcn},
                      {"GB", spec.gb},
                      {"ORD", spec.ord},
                      // Bayer row phase (Start_with_green); 0 for every other op, which ignores it.
                      {"SWG", spec.swg},
                      // The one depth constexpr. Only Op::Bayer can be non-zero -- normalize
                      // refuses every other op at 16U -- but it is passed for all of them so the
                      // key always names the same set, which is what the compile helper checks
                      // (tools/compile_helper.py: `constexprs != the kernel's`).
                      {"U16", n.u16 ? 1 : 0},
                      {"F32", n.f32 ? 1 : 0},
                      {"I0", spec.perm[0]},
                      {"I1", spec.perm[1]},
                      {"I2", spec.perm[2]},
                      {"HR", spec.hrange},
                      {"I3", spec.perm[3]},
                      {"INV", spec.inverse ? 1 : 0},
                      {"OP", static_cast<int>(spec.op)},
                      {"SCN", spec.scn},
                      // The Xyz/Lab/Luv coefficient matrix and the sRGB flag. Passed for EVERY
                      // code, zeros included, so the compile key always names the same constexpr
                      // set -- which is exactly what the helper enforces. They come from the
                      // CODE (coeffs_for), so they are constexprs and not launch arguments.
                      {"SRGB", spec.srgb ? 1 : 0},
                      {"E0", spec.k[0]}, {"E1", spec.k[1]}, {"E2", spec.k[2]},
                      {"E3", spec.k[3]}, {"E4", spec.k[4]}, {"E5", spec.k[5]},
                      {"E6", spec.k[6]}, {"E7", spec.k[7]}, {"E8", spec.k[8]}};
    std::sort(key.constexprs.begin(), key.constexprs.end());

    std::shared_ptr<ModuleRes> mod = moduleFor(key, ctx, gen, st, c.cvt_kernels);
    if (!mod) return st;

    // The two int[256] tables RGB2HSV_b divides by. They are uploaded through the same
    // stream-ordered table cache the resize plans use, so a launch can never read one before
    // its upload completed.
    kf_cvtcolor::HsvTables hv =
        (spec.code == 66 || spec.code == 67) ? kf_cvtcolor::hsv_tables_full() : kf_cvtcolor::hsv_tables();
    std::shared_ptr<TableRes> tab = cvtTablesFor(hv, spec.code, ctx, gen, stream, st);
    if (!tab) return st;

    // The Xyz/Lab/Luv table set, for the codes that read one. XYZ has none (its coefficients are
    // constexprs); the others get the family's tables, and the pointers that family does not use
    // still resolve to something valid inside the same allocation.
    std::shared_ptr<TableRes> ltab = labluvTablesFor(spec.code, ctx, gen, stream, st);
    if (st.code == kf_resize_rt::Code::Error) return st;

    std::map<std::string, Arg> a;
    auto ptr = [&](const char* k, CUdeviceptr v) { Arg x; x.kind = 'p'; x.p = v; a[k] = x; };
    auto i32 = [&](const char* k, long long v) { Arg x; x.kind = 'i'; x.i = static_cast<int>(v); a[k] = x; };
    ptr("src", req.src.ptr);
    ptr("dst", req.dst.ptr);
    ptr("sdiv", tab->at("sdiv"));
    ptr("hdiv", tab->at("hdiv"));
    // The three typed table pointers. For a family that reads none of them (every op but
    // Xyz/Lab/Luv, plus Xyz which has no table at all) these resolve to the four-byte dummy the
    // small table set carries; the per-code constexprs mean the dead branches never load from it.
    const CUdeviceptr t32 = ltab ? ltab->at("t32") : tab->at("t32");
    const CUdeviceptr t16 = ltab ? ltab->at("t16") : tab->at("t16");
    const CUdeviceptr t64 = ltab ? ltab->at("t64") : tab->at("t64");
    ptr("t32", t32);
    ptr("t16", t16);
    ptr("t64", t64);
    // DeviceView::step is the row pitch in BYTES. The kernel indexes a TYPED pointer, so the
    // strides it receives are in ELEMENTS; for CV_8U the two are the same number, which is why
    // this division only appeared when Bayer made 16U reachable. normalize() has already
    // checked that step is a multiple of the element size, so the division is exact.
    i32("cols", n.cols);
    i32("src_rs", static_cast<long long>(req.src.step / kf_resize_rt::elem_size(req.src.type)));
    i32("dst_rs", static_cast<long long>(req.dst.step / kf_resize_rt::elem_size(req.dst.type)));
    // Where the chroma plane starts, in elements from `src`: the Y plane is `n.rows` tall. For a
    // code with no chroma plane the value is unused. It uses the DESTINATION height, which for a
    // 4:2:0 decode is the source height * 2/3 -- using src.rows would read 1.5x too far into the
    // buffer, and that is a silent wrong image rather than a crash.
    i32("uv_rs", static_cast<long long>(n.rows) *
                 static_cast<long long>(req.src.step / kf_resize_rt::elem_size(req.src.type)));

    unsigned grid[3];
    kf_cvtcolor::grid(n, key.cfg.block_w, grid);
    st = launch(*mod, a, grid, stream);
    if (!st.ok()) return st;

    // Under capture the module and tables are pinned to the graph's lifetime instead of being
    // left to the caches: see finishLaunch.
    return finishLaunch(s, ctx, gen, stream, mod, tab, spec.name, key.text(), mod->dir, ltab);
}

// ----------------------------------------------------------------- warpAffine

// Same shape as cvtColor: the plan half (validation, the sampling matrix, the key) lives in
// warpaffine.cpp and this publishes the key and launches.
Status warpAffine(const kf_warpaffine::Request& req, const ExecContext& ex)
{
    kf_warpaffine::Normalized n;
    Status st = kf_warpaffine::normalize(req, n);
    if (!st.ok()) return st;

    const Config& c = config();
    if (!c.error.empty()) return error(Reason::ToolchainUnavailable, c.error);
    if (!c.wa_error.empty()) return error(Reason::ToolchainUnavailable, c.wa_error);
    CUcontext ctx;
    std::uint64_t gen = 0;
    st = currentContext(ex.ctx.device, ctx, gen);
    if (!st.ok()) return st;
    const CUstream stream = reinterpret_cast<CUstream>(ex.stream);
    State& s = S();
    std::vector<std::shared_ptr<void>> released;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        reapLocked(s, released, capturing(stream));
    }

    CUdevice dev;
    cuCtxGetDevice(&dev);
    int major = 0, minor = 0, warp = 32;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);

    const kf_warpaffine::Mode wmode = n.mode;
    const char* wkernel = wmode == kf_warpaffine::Mode::Nearest ? "_warp_nearest"
                        : (wmode == kf_warpaffine::Mode::Linear ? "_warp_linear" : "_warp_cubic");
    const char* wroute = wmode == kf_warpaffine::Mode::Nearest ? "nearest"
                       : (wmode == kf_warpaffine::Mode::Linear ? "linear" : "cubic");

    KernelKey key;
    key.kernel = wkernel;
    key.signature = kf_warpaffine::signature(n);
    key.cfg.block_h = n.block_h;
    key.cfg.block_w = n.block_w;
    key.cfg.num_warps = n.num_warps;
    key.cfg.num_stages = 3;
    key.cfg.ry = 1;
    key.target.backend = "cuda";
    key.target.arch = major * 10 + minor;
    key.target.warp_size = warp;
    key.toolchain.compiler = c.compiler;
    key.toolchain.source_sha256 = c.wa_source_sha256;
    // U8/U16 only exist on the kernels that convert their loads (LINEAR and CUBIC);
    // `_warp_nearest` copies the source element unchanged, so its dtype is carried by the
    // pointer type alone. The key must list exactly the constexprs each kernel actually
    // declares -- the compile helper rejects a mismatch.
    key.constexprs = {{"BLOCK_H", n.block_h},
                      {"BLOCK_W", n.block_w},
                      {"BORDER", n.border_mode_code},
                      {"BORDER_X", n.border_x},
                      {"BORDER_Y", n.border_y},
                      // TRANSPARENT is a separate constexpr rather than folded into BORDER so the
                      // kernels keep their existing border switch and only add the store mask.
                      {"TRANSPARENT", n.border_mode_code == 5 ? 1 : 0},
                      {"cn", n.channels}};
    // All three kernels take FUSED. CUBIC's *coordinate* has no fma (that is what makes it
    // width-invariant), but its bicubic weights and its accumulation are v_fma on the CPU, so
    // it follows the same switch as the other two.
    key.constexprs.push_back({"FUSED", n.fused ? 1 : 0});
    if (n.mode != kf_warpaffine::Mode::Nearest) {
        key.constexprs.push_back({"U8", n.u8 ? 1 : 0});
        key.constexprs.push_back({"U16", n.u16 ? 1 : 0});
    }
    std::sort(key.constexprs.begin(), key.constexprs.end());

    std::shared_ptr<ModuleRes> mod = moduleFor(key, ctx, gen, st, c.wa_kernels);
    if (!mod) return st;

    std::map<std::string, Arg> a;
    auto ptr = [&](const char* k, CUdeviceptr v) { Arg x; x.kind = 'p'; x.p = v; a[k] = x; };
    auto i32 = [&](const char* k, long long v) { Arg x; x.kind = 'i'; x.i = static_cast<int>(v); a[k] = x; };
    auto f32 = [&](const char* k, double v) { Arg x; x.kind = 'f'; x.f = static_cast<float>(v); a[k] = x; };
    ptr("src", req.src.ptr);
    ptr("dst", req.dst.ptr);
    f32("M00", n.A[0]); f32("M01", n.A[1]); f32("M02", n.A[2]);
    f32("M10", n.A[3]); f32("M11", n.A[4]); f32("M12", n.A[5]);
    i32("src_h", n.src_h);
    i32("src_w", n.src_w);
    i32("dst_h", n.dst_h);
    i32("dst_w", n.dst_w);
    // Row pitch in ELEMENTS. A GpuMat's step is padded for alignment, so the kernel cannot
    // recompute the pitch from the width: doing that reads the wrong row on every row but the
    // first, which is the kind of bug that looks like a wrong matrix.
    i32("src_rs", static_cast<long long>(req.src.step / kf_resize_rt::elem_size(req.src.type)));
    i32("dst_rs", static_cast<long long>(req.dst.step / kf_resize_rt::elem_size(req.dst.type)));
    // The Scalar, in the source dtype. A runtime argument rather than a constexpr: a CV_32F
    // border of 0.5 must survive exactly, and as a constexpr it would be truncated. For CV_8U
    // the value is what saturate_cast<uchar> makes of it, which is what the CPU fills.
    for (int i = 0; i < 4; ++i) {
        Arg x; x.kind = 'f'; x.f = static_cast<float>(n.border[i]);
        a[std::string("bv") + std::to_string(i)] = x;
    }

    unsigned g[3];
    kf_warpaffine::grid(n, g);
    st = launch(*mod, a, g, stream);
    if (!st.ok()) return st;

    return finishLaunch(s, ctx, gen, stream, mod, nullptr, wroute, key.text(), mod->dir);
}

}  // namespace runtime

namespace detail {

RuntimeStats runtimeStats()
{
    State& s = S();
    std::lock_guard<std::mutex> lk(s.mu);
    RuntimeStats r = s.st;
    r.module_frees = g_module_frees.load();
    r.table_frees = g_table_frees.load();
    return r;
}

void releaseCapturePins()
{
    State& s = S();
    std::vector<std::shared_ptr<void>> released;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        for (CapturePin& p : s.capture_pins)
            released.insert(released.end(), p.keep.begin(), p.keep.end());
        s.capture_pins.clear();
        s.st.capture_pin_holds = 0;
    }
    // Destroyed here, outside the lock: unloading a module drains the context.
}

void resetRuntimeStats()
{
    State& s = S();
    std::lock_guard<std::mutex> lk(s.mu);
    s.st = RuntimeStats();
    s.st.pending = static_cast<long long>(s.pending.size());
    g_module_frees = 0;
    g_table_frees = 0;
}

void setCacheLimits(int modules, int tables)
{
    State& s = S();
    std::vector<std::shared_ptr<ModuleRes>> em;
    std::vector<std::shared_ptr<TableRes>> et;
    std::lock_guard<std::mutex> lk(s.mu);
    s.modules.cap = static_cast<std::size_t>(std::max(1, modules));
    s.tables.cap = static_cast<std::size_t>(std::max(1, tables));
    s.modules.shrink(em);
    s.tables.shrink(et);
    countEvicted(em, s.st.module_evictions, s.st.deferred);
    countEvicted(et, s.st.table_evictions, s.st.deferred);
}

void reapCompleted()
{
    State& s = S();
    std::vector<std::shared_ptr<void>> released;
    std::lock_guard<std::mutex> lk(s.mu);
    reapLocked(s, released);
}

void setPoisonNewTables(bool on)
{
    State& s = S();
    std::lock_guard<std::mutex> lk(s.mu);
    s.poison = on;
}

std::string runtimeConfig()
{
    const Config& c = config();
    return "resources=" + c.resources + ";kernels=" + c.kernels + ";helper=" + c.helper + ";cache=" + c.cache +
           ";python=" + c.python + ";pythonpath=" + c.pythonpath + ";compiler=" + c.compiler +
           ";source_sha256=" + c.source_sha256 + (c.error.empty() ? std::string() : ";error=" + c.error);
}

}  // namespace detail

}}  // namespace cv::triton
