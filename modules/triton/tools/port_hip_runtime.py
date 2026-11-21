#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

# Copyright 2026, The FlagOS Contributors.
"""Derive src/runtime/hip_runtime.cpp from src/runtime/cuda_runtime.cpp.

    port_hip_runtime.py            write src/runtime/hip_runtime.cpp
    port_hip_runtime.py --check    exit 1 if the checked-in file is not what this would write

The CUDA file is the common source. This generator owns the HIP differences:
driver signatures, target queries, context identification and guarded colour dispatch.
Checked replacement anchors/counts and the final CUDA-token scan detect source changes
that need porting updates. Run --check after editing either owner; do not edit the
generated runtime directly. This check proves reproducible derivation, not GPU behavior.
"""
import re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
src = (HERE / "src/runtime/cuda_runtime.cpp").read_text()
out = src

def sub(old, new, count=None):
    global out
    n = out.count(old)
    if count is not None and n != count:
        sys.exit(f"expected {count} x {old!r}, found {n}")
    if n == 0:
        sys.exit(f"pattern not found: {old!r}")
    out = out.replace(old, new)

# ---- 1. header: what this file is, and why it differs ----------------------------------
sub('// Runtime JIT for the CUDA driver API. See runtime.hpp for execution and ownership rules.\n',
'''// Runtime JIT for the HIP driver API (ROCm/DTK). See runtime.hpp for execution and ownership rules.
//
// GENERATED from cuda_runtime.cpp by tools/port_hip_runtime.py -- do not edit; edit the CUDA
// file and re-run the port. Exactly one of the two is compiled into a build (CMakeLists.txt).
//
// What differs from the CUDA file, and why -- everything else is the same code:
//   * Target: backend "hip", numeric gfx architecture and warp size queried from the device.
//     BW1000 uses gfx936/wave64; W7900D uses gfx1100/wave32. Letter-suffix targets such as
//     gfx90a cannot be represented by this key and are rejected.
//   * Kernels: kernels/<op>_cpu_hip.py where the CUDA source calls a libdevice primitive
//     the HIP compiler lacks; tools/make_kernels_cpu_hip.py derives resize, cvtColor
//     and warpAffine sources and rejects stale copies.
//   * Binary: manifest format "hsaco", loaded with hipModuleLoadData.
//   * Compile helper environment: ROCM_PATH is forwarded (the hip compiler finds clang and its
//     device libraries through it), and the GPU is hidden from the helper with the HIP and ROCR
//     visibility variables as CUDA_VISIBLE_DEVICES= does for cuda.
//   * Dynamic shared memory above the 64 KiB LDS is refused rather than requested through an
//     attribute call, which HIP defines for host function pointers, not module functions.
//   * Context identity uses DTK's hipCtxGetId when available, otherwise retained primary
//     contexts on ROCm. Deprecation warnings for the required hipCtx* API are local to this file.
''', 1)
sub('#include <cuda.h>\n', '#include <hip/hip_runtime_api.h>\n', 1)
sub('#include <sys/stat.h>\n', '#include <sys/stat.h>\n\n#pragma GCC diagnostic ignored "-Wdeprecated-declarations"\n', 1)
sub('#include <algorithm>\n', '#include <algorithm>\n#include <cctype>\n', 1)

# ---- 2. configuration: hip kernels where needed, ROCM_PATH, helper env -----------------
sub('    std::string resources, kernels, helper, cache, python, pythonpath, compiler, source_sha256, error;\n',
    '    std::string resources, kernels, helper, cache, python, pythonpath, compiler, rocm_path, source_sha256, error;\n', 1)
sub('    return isFile(d + "/kernels/resize_cpu.py") && isFile(d + "/tools/compile_helper.py");\n',
    '    return isFile(d + "/kernels/resize_cpu_hip.py") && isFile(d + "/tools/compile_helper.py");\n', 1)
sub('        k.kernels = r + "/kernels/resize_cpu.py";\n', '        k.kernels = r + "/kernels/resize_cpu_hip.py";\n', 1)
sub('            k.error = "kernels/resize_cpu.py and tools/compile_helper.py not found under " + r;\n',
    '            k.error = "kernels/resize_cpu_hip.py and tools/compile_helper.py not found under " + r;\n', 1)
# Float cvtColor uses explicit libdevice rounding; derive its HIP source.
sub('        k.cvt_kernels = r + "/kernels/cvtcolor_cpu.py";\n',
    '        k.cvt_kernels = r + "/kernels/cvtcolor_cpu_hip.py";\n', 1)
sub('        k.wa_kernels = r + "/kernels/warpaffine_cpu.py";\n',
    '        k.wa_kernels = r + "/kernels/warpaffine_cpu_hip.py";\n', 1)
sub('        k.compiler = envStr("OPENCV_TRITON_COMPILER");\n',
    '        k.rocm_path = envStr("ROCM_PATH");\n'
    '        if (k.rocm_path.empty()) k.rocm_path = OPENCV_TRITON_ROCM_PATH_DEFAULT;\n'
    '        k.compiler = envStr("OPENCV_TRITON_COMPILER");\n', 1)
sub('             "PYTHONPATH=" + c.pythonpath, "CUDA_VISIBLE_DEVICES=", "LC_ALL=C"};   // the helper needs no device\n',
    '             "PYTHONPATH=" + c.pythonpath,\n'
    '             "TMPDIR=" + c.cache,\n'
    '             "ROCM_PATH=" + c.rocm_path,   // the hip compiler finds clang and its device libraries through this\n'
    '             "HIP_VISIBLE_DEVICES=", "ROCR_VISIBLE_DEVICES=", "CUDA_VISIBLE_DEVICES=", "LC_ALL=C"};   // no device\n', 1)

# ---- 3. error text -------------------------------------------------------------------
sub('''// --------------------------------------------------------------------- CUDA

std::string cuErr(CUresult r)
{
    const char* s = nullptr;
    cuGetErrorName(r, &s);
    return s ? std::string(s) : ("CUresult " + std::to_string(static_cast<int>(r)));
}''', '''// ---------------------------------------------------------------------- HIP

std::string hipErr(hipError_t r)
{
    const char* s = hipGetErrorName(r);
    return s ? std::string(s) : ("hipError_t " + std::to_string(static_cast<int>(r)));
}''', 1)
out = re.sub(r'\bcuErr\(', 'hipErr(', out)

# ---- 4. calls whose signature differs (before the token renames) -----------------------
# CUdeviceptr is an integer, hipDeviceptr_t a void*: the resources keep integers, the calls cast.
sub('    CUdeviceptr base = 0;\n', '    std::uintptr_t base = 0;\n', None)
sub('const CUresult result = stream_ordered ? cuMemFreeAsync(base, nullptr) : cuMemFree(base);',
    'const hipError_t result = stream_ordered ? hipFreeAsync(reinterpret_cast<void*>(base), nullptr) : hipFree(reinterpret_cast<void*>(base));', 1)
alloc_old = '''    CUresult r = t->stream_ordered ? cuMemAllocAsync(&t->base, t->bytes, stream) : cuMemAlloc(&t->base, t->bytes);
    if (r == CUDA_SUCCESS && poison) r = cuMemsetD8Async(t->base, 0xFF, t->bytes, stream);
    if (r == CUDA_SUCCESS) { t->upload_started = true; r = cuMemcpyHtoDAsync(t->base, t->host.data(), t->bytes, stream); }
    if (r == CUDA_SUCCESS) r = cuEventCreate(&t->uploaded, CU_EVENT_DISABLE_TIMING);
    if (r == CUDA_SUCCESS) { r = cuEventRecord(t->uploaded, stream); t->upload_recorded = r == CUDA_SUCCESS; }'''
alloc_new = '''    void* dptr = nullptr;
    hipError_t r = t->stream_ordered ? hipMallocAsync(&dptr, t->bytes, stream) : hipMalloc(&dptr, t->bytes);
    t->base = reinterpret_cast<std::uintptr_t>(dptr);
    if (r == hipSuccess && poison) r = hipMemsetD8Async(dptr, 0xFF, t->bytes, stream);
    if (r == hipSuccess) { t->upload_started = true; r = hipMemcpyHtoDAsync(dptr, t->host.data(), t->bytes, stream); }
    if (r == hipSuccess) r = hipEventCreateWithFlags(&t->uploaded, hipEventDisableTiming);
    if (r == hipSuccess) { r = hipEventRecord(t->uploaded, stream); t->upload_recorded = r == hipSuccess; }'''
n_alloc = out.count(alloc_old)
if n_alloc < 2: sys.exit(f"table-upload block: expected >=2, found {n_alloc}")
out = out.replace(alloc_old, alloc_new)

copy_old = re.compile(r'''        CUDA_MEMCPY2D cp;
        std::memset\(&cp, 0, sizeof cp\);
        cp\.srcMemoryType = CU_MEMORYTYPE_DEVICE; cp\.srcDevice = (?P<s>[\w.]+)\.ptr; cp\.srcPitch = (?P=s)\.step;
        cp\.dstMemoryType = CU_MEMORYTYPE_DEVICE; cp\.dstDevice = (?P<d>[\w.]+)\.ptr; cp\.dstPitch = (?P=d)\.step;
        cp\.WidthInBytes = (?P<w>[^;]+); cp\.Height = (?P<h>[^;]+);
        const CUresult r = cuMemcpy2DAsync\(&cp, stream\);''')
def copy_new(m):
    return (f'''        const hipError_t r = hipMemcpy2DAsync(reinterpret_cast<void*>({m["d"]}.ptr), {m["d"]}.step,
                                              reinterpret_cast<const void*>({m["s"]}.ptr), {m["s"]}.step,
                                              {m["w"]}, {m["h"]},
                                              hipMemcpyDeviceToDevice, stream);''')
out, n_copy = copy_old.subn(copy_new, out)
if n_copy < 1: sys.exit("2-D copy block not found")

load_old = '''    CUresult r = cuModuleLoadData(&m->mod, bin.data());
    if (r == CUDA_SUCCESS) r = cuModuleGetFunction(&m->fn, m->mod, man.function.c_str());
    if (r == CUDA_SUCCESS && man.shared_bytes > 48 * 1024)
        r = cuFuncSetAttribute(m->fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, man.shared_bytes);'''
load_new = '''    if (man.format != "hsaco") { st = error(Reason::ManifestInvalid, "binary format '" + man.format + "' is not hsaco"); return nullptr; }
    if (man.shared_bytes > 64 * 1024)
    {
        st = error(Reason::LoadFailed, "kernel asks for " + std::to_string(man.shared_bytes) + " bytes of dynamic shared memory; the LDS is 64 KiB");
        return nullptr;
    }
    hipError_t r = hipModuleLoadData(&m->mod, bin.data());
    if (r == hipSuccess) r = hipModuleGetFunction(&m->fn, m->mod, man.function.c_str());'''
sub(load_old, load_new, 1)

sub('''    const CUresult r = cuLaunchKernel(m.fn, grid[0], grid[1], grid[2], static_cast<unsigned>(m.man.threads_per_block), 1, 1,
                                      static_cast<unsigned>(m.man.shared_bytes), stream, params.data(), nullptr);
    if (r != CUDA_SUCCESS) return error(Reason::LaunchFailed, "cuLaunchKernel: " + hipErr(r));''',
'''    const hipError_t r = hipModuleLaunchKernel(m.fn, grid[0], grid[1], grid[2], static_cast<unsigned>(m.man.threads_per_block), 1, 1,
                                               static_cast<unsigned>(m.man.shared_bytes), stream, params.data(), nullptr);
    if (r != hipSuccess) return error(Reason::LaunchFailed, "hipModuleLaunchKernel: " + hipErr(r));''', 1)

sub('''    CUstreamCaptureStatus st = CU_STREAM_CAPTURE_STATUS_NONE;
    return cuStreamIsCapturing(stream, &st) == CUDA_SUCCESS && st != CU_STREAM_CAPTURE_STATUS_NONE;''',
'''    hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
    return hipStreamIsCapturing(stream, &st) == hipSuccess && st != hipStreamCaptureStatusNone;''', 1)

# Keep HIP's checked DevInfo query and existing cache; CUDA's checked helper
# has no HIP role. Restore the three query blocks before their normal translation.
out = re.sub(r"// Validated CUDA target facts\..*?// End validated CUDA target facts\.\n\n", "", out, flags=re.S)
out = out.replace('    int major = 0, minor = 0, warp = 32;\n    st = deviceAttributes(ex.ctx.device, major, minor, warp);\n    if (!st.ok())\n        return st;\n', '    CUdevice dev;\n    cuCtxGetDevice(&dev);\n    int major = 0, minor = 0, warp = 32;\n    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);\n    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);\n    cuDeviceGetAttribute(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);\n')

# ---- 5. target identity: three identical query blocks -> one cached device query --------
tgt = re.compile(r'''    CUdevice dev;
    cuCtxGetDevice\(&dev\);
    int major = 0, minor = 0, warp = 32;
    cuDeviceGetAttribute\(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev\);
    cuDeviceGetAttribute\(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev\);
    cuDeviceGetAttribute\(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev\);
''')
tgt_new = '''    DevInfo dev;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        auto it = s.devices.find(ex.ctx.device);
        if (it == s.devices.end()) it = s.devices.emplace(ex.ctx.device, queryDevice(ex.ctx.device)).first;
        dev = it->second;
    }
    if (!dev.ok) return error(Reason::ToolchainUnavailable, dev.why);
'''
out, n_tgt = tgt.subn(tgt_new, out)
if n_tgt != 3: sys.exit(f"target query blocks: expected 3, found {n_tgt}")
out, n_a = re.subn(r'(\b(?:key\.)?target)\.backend = "cuda";\n(\s+)\1\.arch = major \* 10 \+ minor;\n\s+\1\.warp_size = warp;',
                   lambda m: f'{m[1]}.backend = "hip";\n{m[2]}{m[1]}.arch = dev.arch;\n{m[2]}{m[1]}.warp_size = dev.warp;', out)
if n_a != 3: sys.exit(f"target assignments: expected 3, found {n_a}")

# DevInfo + its query live beside State, which owns the per-device cache.
sub('''struct State
{''', '''// The device facts the compile key names, read once per ordinal: the gfx number (the key's
// integer arch) and the wave size.
struct DevInfo
{
    bool ok = false;
    int arch = 0;                // numeric gfx suffix, e.g. 936 or 1100
    int warp = 64;
    std::string arch_name, why;
};

DevInfo queryDevice(int device)
{
    DevInfo d;
    hipDeviceProp_t p;
    std::memset(&p, 0, sizeof p);
    const hipError_t r = hipGetDeviceProperties(&p, device);
    if (r != hipSuccess) { d.why = "hipGetDeviceProperties: " + hipErr(r); return d; }
    d.arch_name = p.gcnArchName;
    d.warp = p.warpSize > 0 ? p.warpSize : 64;
    const std::string& n = d.arch_name;
    std::size_t i = 3;
    long v = 0;
    if (n.compare(0, 3, "gfx") != 0) { d.why = "unrecognised architecture name '" + n + "'"; return d; }
    for (; i < n.size() && std::isdigit(static_cast<unsigned char>(n[i])); ++i) v = v * 10 + (n[i] - '0');
    if (i == 3 || (i < n.size() && n[i] != ':'))
    {
        d.why = "architecture '" + n + "' has no integer gfx number, so the compile key cannot name it";
        return d;
    }
    d.arch = static_cast<int>(v);
    d.ok = true;
    return d;
}

struct State
{
    std::map<int, DevInfo> devices;''', 1)

# ROCm does not provide DTK's hipCtxGetId extension. Preserve it when exported;
# otherwise support retained primary contexts only. No pointer-derived fake
# generation: the retained context is pinned for this process's cache lifetime.
# Device reset and primary-context reset are unsupported while this runtime is in use.
context_helper = r'''
hipError_t contextGeneration(hipCtx_t context, unsigned long long* generation, bool registerPrimary = true)
{
    using GetId = hipError_t (*)(hipCtx_t, unsigned long long*);
    static const GetId get_id = reinterpret_cast<GetId>(dlsym(RTLD_DEFAULT, "hipCtxGetId"));
    if (get_id) return get_id(context, generation);
    if (!context || !generation) return hipErrorInvalidContext;
    struct Primary { hipCtx_t context; int device; unsigned long long generation; };
    struct Registry { std::mutex mutex; std::vector<Primary> primaries; };
    static auto* registry = new Registry();
    std::lock_guard<std::mutex> lock(registry->mutex);
    for (const auto& primary : registry->primaries)
        if (primary.context == context)
        {
            unsigned int flags = 0; int active = 0;
            const hipError_t result = hipDevicePrimaryCtxGetState(primary.device, &flags, &active);
            if (result != hipSuccess) return result;
            if (!active) return hipErrorInvalidContext;
            *generation = primary.generation;
            return hipSuccess;
        }
    // Cleanup must not register/retain a new context or switch a destroyed one.
    if (!registerPrimary) return hipErrorInvalidValue;
    hipCtx_t previous = nullptr;
    hipError_t result = hipCtxGetCurrent(&previous);
    if (result != hipSuccess) return result;
    if (previous != context) return hipErrorInvalidContext;
    hipDevice_t device;
    result = hipCtxGetDevice(&device);
    if (result != hipSuccess) return result;
    hipCtx_t primary = nullptr;
    result = primaryContext(static_cast<int>(device), primary);
    // ROCm/DTK retain can change the current context: restore even when ownership
    // acquisition succeeded but this request names a distinct private context.
    const hipError_t restored = hipCtxSetCurrent(previous);
    if (result != hipSuccess) return result;
    if (restored != hipSuccess) return restored;
    if (context != primary) return hipErrorInvalidContext;
    const auto id = static_cast<unsigned long long>(registry->primaries.size()) + 1;
    registry->primaries.push_back({primary, static_cast<int>(device), id});
    *generation = id;
    return hipSuccess;
}

'''
# Declare primaryContext before context identification so hipCtxGetId and the
# primary-context fallback share one process-lifetime reference per device.
sub('ContextLife contextLife(CUcontext c, std::uint64_t gen)\n',
    'CUresult primaryContext(int device, CUcontext& context);\n' + context_helper +
    'ContextLife contextLife(CUcontext c, std::uint64_t gen)\n', 1)
sub('cuCtxGetId(c, &id)', 'contextGeneration(c, &id, false)', 1)
sub('cuCtxGetId(ctx, &id)', 'contextGeneration(ctx, &id)', 1)

# ---- 6. plain token renames -----------------------------------------------------------
pairs = [
    ("CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED", "hipErrorStreamCaptureUnsupported"),
    ("CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED", "hipDeviceAttributeMemoryPoolsSupported"),
    ("CU_EVENT_DISABLE_TIMING", "hipEventDisableTiming"),
    ("CUDA_ERROR_NOT_READY", "hipErrorNotReady"),
    ("CU_STREAM_CAPTURE_STATUS_NONE", "hipStreamCaptureStatusNone"),
    ("CUstreamCaptureStatus", "hipStreamCaptureStatus"),
    ("cuStreamIsCapturing", "hipStreamIsCapturing"),
    ("CUDA_ERROR_INVALID_CONTEXT", "hipErrorInvalidContext"),
    ("CUDA_ERROR_CONTEXT_IS_DESTROYED", "hipErrorContextIsDestroyed"),
    ("CUDA_ERROR_DEINITIALIZED", "hipErrorDeinitialized"),
    ("CUDA_SUCCESS", "hipSuccess"),
    ("CUcontext", "hipCtx_t"), ("CUstream", "hipStream_t"), ("CUresult", "hipError_t"),
    ("CUdevice", "hipDevice_t"), ("CUevent", "hipEvent_t"), ("CUmodule", "hipModule_t"),
    ("CUfunction", "hipFunction_t"), ("CUdeviceptr", "std::uintptr_t"),
    ("cuDevicePrimaryCtxRetain", "hipDevicePrimaryCtxRetain"),
    ("cuDeviceGetAttribute", "hipDeviceGetAttribute"), ("cuDeviceGetCount", "hipGetDeviceCount"),
    ("cuDeviceGet", "hipDeviceGet"), ("cuCtxGetCurrent", "hipCtxGetCurrent"),
    ("cuCtxSetCurrent", "hipCtxSetCurrent"), ("cuCtxPushCurrent", "hipCtxPushCurrent"),
    ("cuCtxPopCurrent", "hipCtxPopCurrent"), ("cuCtxGetDevice", "hipCtxGetDevice"),
    ("cuCtxGetId", "hipCtxGetId"), ("cuStreamWaitEvent", "hipStreamWaitEvent"),
    ("cuEventRecord", "hipEventRecord"), ("cuEventQuery", "hipEventQuery"),
    ("cuEventCreate", "hipEventCreateWithFlags"), ("cuEventDestroy", "hipEventDestroy"),
    ("cuModuleUnload", "hipModuleUnload"), ("cuInit", "hipInit"),
]
for a, b in pairs:
    out = re.sub(rf"\b{a}\b", b, out)
# messages that name the API
out = out.replace('"cuInit failed"', '"hipInit failed"').replace('"hipCtxGetId failed"', '"hipCtxGetId failed"')
out = out.replace('return "cuda-jit";', 'return "hip-jit";')
out = out.replace('// never destroyed: process exit may follow CUDA teardown', '// never destroyed: process exit may follow HIP teardown')

# Driver API spellings only: cuXxx( calls, CU_ macros, CUDA_ codes, CUxxx types. A bare
# all-caps word such as CUBIC (an interpolation) is not one.
# HIP declares these [[nodiscard]]; the CUDA file ignores their results on purpose (a
# best-effort query, a teardown call whose failure has no recovery). Say so with (void),
# rather than silencing -Wunused-result for the whole file.
for call in ("hipCtxGetCurrent(", "hipCtxPopCurrent(", "hipModuleUnload(", "hipStreamWaitEvent(nullptr",
             "hipEventDestroy(", "hipDeviceGetAttribute("):
    fixed = []
    for line in out.split("\n"):
        i = line.find(call)
        while i != -1:
            before = line[:i].rstrip()
            # A bare statement: the call starts the line, or follows `{`, `;` or an if-condition's `)`.
            if (before == "" or before.endswith(("{", ";", ")"))) and not before.endswith("(void)"):
                line = line[:i] + "(void)" + line[i:]
                i += len("(void)")
            i = line.find(call, i + len(call))
        fixed.append(line)
    out = "\n".join(fixed)
left = sorted(set(re.findall(r"\bcu[A-Z][A-Za-z0-9_]*\(|\bCU_[A-Z_]+\b|\bCUDA_[A-Z_]+\b|\bCU[a-z][A-Za-z]*\b", out)))
# CUDA_VISIBLE_DEVICES is intentionally forwarded (empty) and comments may name CUDA errors.
code_left = [t for t in left if t != "CUDA_VISIBLE_DEVICES" and
             any(t in line and not line.lstrip().startswith("//") for line in out.splitlines())]
if code_left:
    sys.exit(f"CUDA tokens left in code: {code_left}")

# Candidate packed paths keep the same alignment and type admission as CUDA.
sub('std::string(name()) == "cuda-jit" && !n.u16',
    'dev.arch == 936 && (spec.code == 0 || spec.code == 4) && !n.u16', 1)
sub('std::string(name()) == "cuda-jit" && n.u8',
    'dev.arch == 936 && n.dst_h >= 64 && n.dst_w >= 256 && '
    'static_cast<long long>(n.dst_h) * n.dst_w >= 65536 && n.u8', 1)
# Keep HIP interpolation and nearest tiles independent of CUDA defaults.

# HIP colour vectorization keeps the CUDA runtime's generic route unchanged.
sub('''
    const std::string keyText = key.text();
    std::shared_ptr<ModuleRes> mod = moduleFor(key, keyText, ctx, gen, st, c.cvt_kernels);''',
'''    const bool packed = dev.arch == 936 && !n.u16 && !n.f32 && n.cols >= 1024 && n.cols % 4 == 0 &&
        req.src.ptr % 4 == 0 && req.dst.ptr % 4 == 0 &&
        req.src.step % 4 == 0 && req.dst.step % 4 == 0 &&
        spec.code == 6 && n.n >= (1LL << 19);
    if (packed)
    {
        key.kernel = "_cvtcolor_hip_packed";
        key.signature = "src:*u8, dst:*u8, cols:i32, src_rs:i32, dst_rs:i32";
        key.cfg.block_w = n.n < (1LL << 22) ? 1024 : 512;
        key.cfg.num_warps = n.n < (1LL << 22) ? 4 : 2;
        key.constexprs = {{"BLOCK", key.cfg.block_w}};
    }
    std::sort(key.constexprs.begin(), key.constexprs.end());

    const std::string keyText = key.text();
    std::shared_ptr<ModuleRes> mod = moduleFor(key, keyText, ctx, gen, st, c.cvt_kernels);''', 1)
sub('''    cv_triton_color::grid(n, key.cfg.block_w, grid);''',
'''    cv_triton_color::grid(n, key.cfg.block_w, grid);
    if (packed)
    {
        // A lane covers four complete pixels; row pitches stay independent.
        grid[1] = static_cast<unsigned>((n.cols / 4 + key.cfg.block_w - 1) / key.cfg.block_w);
    }''', 1)

# gfx936 launch defaults preserve explicit schemas and other HIP architectures.
sub("""    const LaunchShape shape = launch_shape(n, LaunchConfig{}, target, tc, 1, &req.src, &req.dst);""",
"""    LaunchConfig requested;
    if (dev.arch == 936 && n.route.kind == ref::ROUTE_CUBIC && n.inv_y > 0.25 &&
        n.dst_h <= n.src_h && n.dst_w <= n.src_w &&
        static_cast<long long>(n.dst_h) * n.dst_w >= 65536)
    {
        // Reuse coefficients and overlapping rows without starving independent programs.
        int rows = 16;
        const auto programs = [&](int height)
        {
            return static_cast<long long>((n.dst_h + height - 1) / height) *
                   ((n.dst_w + 127) / 128);
        };
        while (rows > 4 && programs(rows) < DISPATCH_MIN_PROGRAMS)
            rows /= 2;
        if (programs(rows) >= DISPATCH_MIN_PROGRAMS)
        {
            requested.block_h = rows;
            requested.block_w = 128;
            requested.num_warps = 2;
        }
    }
    LaunchShape shape = launch_shape(n, requested, target, tc, 1, &req.src, &req.dst);
    if (target.arch == 936 && n.route.kind == ref::ROUTE_CUBIC && shape.key.cfg.block_h > 1 &&
        n.inv_y > 0.25 && n.inv_y < 1.0)
        shape.key.kernel = "_resize_cpu_sep_reuse_kernel";
    if (target.arch == 936 && n.route.kind == ref::ROUTE_CUBIC && n.type == ElemType::F32 &&
        n.inv_y >= 1.0 && (static_cast<long long>(n.dst_w) * n.channels) % 4 == 0)
        shape.key.kernel = "_resize_cpu_sep_f32_aligned_kernel";""", 1)

sub("""#include "resize_plan.hpp"
""",
"""#include "resize_plan.hpp"
#include "dispatch_table.hpp"
""", 1)

# All colour constexpr initializers are already in canonical lexical order.
# Values cannot affect order because names are unique; retain identical key bytes.
sub("""    std::sort(key.constexprs.begin(), key.constexprs.end());

    const std::string keyText = key.text();
    std::shared_ptr<ModuleRes> mod = moduleFor(key, keyText, ctx, gen, st, c.cvt_kernels);""",
"""    const std::string keyText = key.text();
    std::shared_ptr<ModuleRes> mod = moduleFor(key, keyText, ctx, gen, st, c.cvt_kernels);""", 1)

sub("""    const std::string keyText = key.text();
    std::shared_ptr<ModuleRes> mod = moduleFor(key, keyText, ctx, gen, st, c.cvt_kernels);""",
"""    if (dev.arch == 936 && !n.u16 && !n.f32 && n.n >= 65536 && n.cols >= 512 && (spec.code == 10 || spec.code == 82) && !cudaPacked && !packed)
    {
        key.cfg.block_w = 512;
        key.cfg.num_warps = 4;
        for (auto &constant : key.constexprs)
            if (constant.first == "BLOCK") constant.second = 512;
    }
    if (dev.arch == 936 && spec.code == 10 && !n.u16 && !n.f32 && n.n >= 65536 && n.cols % 4 == 0 &&
        req.src.ptr % 4 == 0 && req.dst.ptr % 4 == 0 && req.src.step % 4 == 0 && req.dst.step % 4 == 0)
    {
        key.kernel = "_cvtcolor_hip_bgra_gray";
        key.signature = "src:*u8, dst:*u8, cols:i32, src_rs:i32, dst_rs:i32";
        key.cfg.block_w = 256;
        key.cfg.num_warps = 4;
        key.constexprs = {{"BLOCK", key.cfg.block_w}};
    }
    const std::string keyText = key.text();
    std::shared_ptr<ModuleRes> mod = moduleFor(key, keyText, ctx, gen, st, c.cvt_kernels);""", 1)

sub("""    const std::string keyText = key.text();
    std::shared_ptr<ModuleRes> mod = moduleFor(key, keyText, ctx, gen, st, c.cvt_kernels);""",
"""    if (dev.arch == 936 && spec.code == 0 && n.f32 && n.n >= 65536 && n.cols >= 256)
    {
        key.kernel = "_cvtcolor_hip_alpha_f32";
        key.signature = "src:*fp32, dst:*fp32, cols:i32, src_rs:i32, dst_rs:i32";
        key.cfg.block_w = 256;
        key.cfg.num_warps = 4;
        key.constexprs = {{"BLOCK", key.cfg.block_w}};
    }
    const std::string keyText = key.text();
    std::shared_ptr<ModuleRes> mod = moduleFor(key, keyText, ctx, gen, st, c.cvt_kernels);""", 1)

sub("""    if (packed)
    {
        // A lane covers four complete pixels; row pitches stay independent.""",
"""    if (packed || key.kernel == "_cvtcolor_hip_bgra_gray")
    {
        // A lane covers four complete pixels; row pitches stay independent.""", 1)

sub("""    const bool packed4 = dev.arch == 936""",
"""    if (dev.arch == 936 && n.channels == 1 && n.mode == cv_triton_warp::Mode::Cubic &&
        std::abs(n.A[3]) >= 0.125 && n.border_mode_code == 1 && n.dst_h >= 64 && n.dst_w >= 256 &&
        static_cast<long long>(n.dst_h) * n.dst_w >= 65536)
    {
        n.block_h = 4;
        n.block_w = 32;
        n.num_warps = 2;
    }
    const bool packed4 = dev.arch == 936""", 1)

sub("""    const char *wroute = wmode == cv_triton_warp::Mode::Nearest""",
"""    // Modulo borders use channel tiles only where their arithmetic cost is offset
    // by cubic tap reuse or four-channel float loads.
    const bool channelBorder = n.border_mode_code == 0 || n.border_mode_code == 5 || n.border_mode_code == 1 ||
        ((n.border_mode_code >= 2 && n.border_mode_code <= 4) &&
         ((n.mode == cv_triton_warp::Mode::Cubic && (n.channels == 4 || req.src.type != ElemType::U16)) ||
          (n.mode == cv_triton_warp::Mode::Linear && n.channels == 4 && req.src.type == ElemType::F32)));
    if (dev.arch == 936 && (n.channels == 3 || (n.channels == 4 && (!packed4 || n.mode == cv_triton_warp::Mode::Cubic || n.border_mode_code != 1))) &&
        n.mode != cv_triton_warp::Mode::Nearest && ((n.channels == 4 && n.border_mode_code == 1) || std::abs(n.A[3]) >= 0.125) &&
        channelBorder && n.dst_w >= 256 && n.dst_h >= 64 &&
        static_cast<long long>(n.dst_w) * n.dst_h >= 65536)
    {
        wkernel = n.mode == cv_triton_warp::Mode::Linear ? "_warp_linear_hip_channels" : "_warp_cubic_hip_channels";
        // Large float C4 frames benefit from keeping two nearby rows in a program.
        const bool twoRows = n.border_mode_code == 1 && n.channels == 4 && req.src.type == ElemType::F32 && wmode == cv_triton_warp::Mode::Linear &&
                             std::abs(n.A[3]) >= 0.125 &&
                             static_cast<long long>(n.src_h) * n.src_w >= (1LL << 22) &&
                             static_cast<long long>(n.dst_h) * n.dst_w >= (1LL << 22);
        n.block_h = twoRows ? 2 : 1;
        n.block_w = twoRows ? 64 : 128;
        n.num_warps = 4;
    }
    const char *wroute = wmode == cv_triton_warp::Mode::Nearest""", 1)

# The conservative byte bound also bounds every valid element offset.
sub("#include <list>", "#include <list>\n#include <limits>", 1)
sub("""    const char *wroute = wmode == cv_triton_warp::Mode::Nearest""",
"""    const std::size_t offsetLimit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (dev.arch == 936 && n.u16 && n.channels == 4 && n.border_mode_code == 1 &&
        std::strcmp(wkernel, "_warp_cubic_hip_channels") == 0 &&
        req.src.step <= offsetLimit / static_cast<std::size_t>(n.src_h) &&
        req.dst.step <= offsetLimit / static_cast<std::size_t>(n.dst_h))
        wkernel = "_warp_cubic_i32_hip_channels";
    const char *wroute = wmode == cv_triton_warp::Mode::Nearest""", 1)

sub("""    const char *wroute = wmode == cv_triton_warp::Mode::Nearest""",
"""    if (dev.arch == 936 && n.u8 && n.channels == 4 && n.border_mode_code == 1 &&
        wmode == cv_triton_warp::Mode::Cubic && n.dst_w >= 256 && n.dst_h >= 64 &&
        static_cast<long long>(n.dst_w) * n.dst_h >= 65536 &&
        req.src.ptr % 4 == 0 && req.dst.ptr % 4 == 0 &&
        req.src.step % 4 == 0 && req.dst.step % 4 == 0 &&
        (n.A[3] == 0.0 || (static_cast<long long>(n.src_w) * n.src_h < (1LL << 22) &&
                          static_cast<long long>(n.dst_w) * n.dst_h < (1LL << 22))))
    {
        wkernel = "_warp_cubic_packed4_u8";
        n.block_h = 1;
        n.block_w = 128;
        n.num_warps = 2;
    }
    const char *wroute = wmode == cv_triton_warp::Mode::Nearest""", 1)

sub("""    const char *wroute = wmode == cv_triton_warp::Mode::Nearest""",
"""    if (dev.arch == 936 && wmode == cv_triton_warp::Mode::Cubic && n.border_mode_code == 1 &&
        ((req.src.type == ElemType::F32 && n.channels == 1) ||
         ((n.u8 || n.u16) && n.channels == 4)) &&
        n.src_w == n.dst_w && n.src_h == n.dst_h && n.src_w < (1 << 23) && n.src_h < (1 << 23) &&
        n.dst_w >= 256 && n.dst_h >= 64 && static_cast<long long>(n.dst_w) * n.dst_h >= 65536 &&
        n.A[0] == 1.0 && n.A[4] == 1.0 && n.A[1] == 0.0 && n.A[3] == 0.0 &&
        std::abs(n.A[2]) == 0.5 && std::abs(n.A[5]) == 0.5)
    {
        if (req.src.type == ElemType::F32 && n.channels == 1)
            wkernel = "_warp_cubic_halfpixel";
        else if (std::strcmp(wkernel, "_warp_cubic_i32_hip_channels") == 0)
            wkernel = "_warp_cubic_i32_hip_channels_halfpixel";
        else if (std::strcmp(wkernel, "_warp_cubic_packed4_u8") == 0)
            wkernel = "_warp_cubic_packed4_u8_halfpixel";
    }
    const char *wroute = wmode == cv_triton_warp::Mode::Nearest""", 1)

# Exact identity nearest and integer interpolation are image copies. Retain validation and
# context/capture/cleanup checks before the shortcut, as for resize copy.
sub("""    // CUDA uses per-mode tiles: interpolation benefits from a""",
"""    const auto copyAligned = [&](std::size_t alignment)
    {
        return req.src.ptr % alignment == 0 && req.dst.ptr % alignment == 0 &&
            req.src.step % alignment == 0 && req.dst.step % alignment == 0 &&
            req.src.row_bytes() % alignment == 0;
    };
    if (dev.arch == 936 &&
        (n.mode == cv_triton_warp::Mode::Nearest ||
         ((n.u8 || n.u16) &&
          (n.mode == cv_triton_warp::Mode::Linear || n.mode == cv_triton_warp::Mode::Cubic))) &&
        n.src_w == n.dst_w && n.src_h == n.dst_h &&
        n.src_w < (1 << 23) && n.src_h < (1 << 23) &&
        n.dst_w >= 256 && n.dst_h >= 64 &&
        static_cast<long long>(n.dst_w) * n.dst_h >= (1 << 18) &&
        n.A[0] == 1 && n.A[1] == 0 && n.A[2] == 0 &&
        n.A[3] == 0 && n.A[4] == 1 && n.A[5] == 0 &&
        (copyAligned(16) || (n.u8 && n.channels == 3 && copyAligned(4))))
    {
        const hipError_t result = hipMemcpy2DAsync(reinterpret_cast<void*>(req.dst.ptr),
            req.dst.step, reinterpret_cast<const void*>(req.src.ptr), req.src.step,
            req.src.row_bytes(), static_cast<std::size_t>(req.src.rows),
            hipMemcpyDeviceToDevice, stream);
        if (result != hipSuccess)
            return error(Reason::LaunchFailed, "identity warp copy: " + hipErr(result));
        std::lock_guard<std::mutex> lock(s.mu);
        ++s.st.copies;
        s.st.last_route = "warp_identity_copy";
        Status ok;
        ok.code = Code::Submitted;
        ok.submitted = true;
        return ok;
    }
    // CUDA uses per-mode tiles: interpolation benefits from a""", 1)

sub("""    const char *wroute = wmode == cv_triton_warp::Mode::Nearest""",
"""    if (dev.arch == 936 && n.u16 && n.channels == 4 && n.border_mode_code == 1 &&
        wmode == cv_triton_warp::Mode::Nearest &&
        req.src.ptr % 8 == 0 && req.dst.ptr % 8 == 0 &&
        req.src.step % 8 == 0 && req.dst.step % 8 == 0 &&
        n.dst_w >= 256 && n.dst_h >= 64 && static_cast<long long>(n.dst_w) * n.dst_h >= 65536)
    {
        wkernel = "_warp_nearest_packed4_u16";
        n.block_h = 1;
        n.block_w = 512;
        n.num_warps = 4;
    }
    const char *wroute = wmode == cv_triton_warp::Mode::Nearest""", 1)

sub("""    const std::string keyText = shape.key.text();
    std::shared_ptr<ModuleRes> mod = moduleFor(shape.key, keyText, ctx, gen, st, c.kernels);""",
"""    std::shared_ptr<TableRes> staticAreaTables;
    if (dev.arch == 936 && n.route.kind == ref::ROUTE_AREA &&
        n.type == ElemType::F32 && n.channels == 1)
    {
        staticAreaTables = tablesFor(n, ctx, gen, stream, st);
        if (!staticAreaTables)
            return st;
        if (staticAreaTables->kx <= 4 && staticAreaTables->ky <= 4)
        {
            shape.key.kernel = "_resize_cpu_area_static_kernel";
            const std::string dynamicTaps = "KX:i32, KY:i32, ";
            const auto offset = shape.key.signature.find(dynamicTaps);
            if (offset == std::string::npos)
                return error(Reason::LaunchFailed, "static AREA signature missing tap counts");
            shape.key.signature.erase(offset, dynamicTaps.size());
            shape.key.constexprs.push_back({"KX", staticAreaTables->kx});
            shape.key.constexprs.push_back({"KY", staticAreaTables->ky});
            std::sort(shape.key.constexprs.begin(), shape.key.constexprs.end());
        }
    }
    const std::string keyText = shape.key.text();
    std::shared_ptr<ModuleRes> mod = moduleFor(shape.key, keyText, ctx, gen, st, c.kernels);""", 1)
sub("""    std::shared_ptr<TableRes> tab;
    if (n.route.kind != ref::ROUTE_AREA_FAST)""",
"""    std::shared_ptr<TableRes> tab = staticAreaTables;
    if (!tab && n.route.kind != ref::ROUTE_AREA_FAST)""", 1)

sub("""    const std::string keyText = key.text();
    std::shared_ptr<ModuleRes> mod = moduleFor(key, keyText, ctx, gen, st, c.cvt_kernels);""",
"""    if (dev.arch == 936 && n.f32 && n.f32_cpu_separate4 &&
        (spec.code == 10 || spec.code == 11) && n.n >= 65536 && n.cols >= 256)
    {
        key.kernel = "_cvtcolor_hip_gray_f32_c4";
        key.signature = "src:*fp32, dst:*fp32, cols:i32, src_rs:i32, dst_rs:i32";
        key.cfg.block_w = 256;
        key.cfg.num_warps = 4;
        key.constexprs = {{"BLOCK", key.cfg.block_w}, {"BLUE", spec.blue}};
    }
    const std::string keyText = key.text();
    std::shared_ptr<ModuleRes> mod = moduleFor(key, keyText, ctx, gen, st, c.cvt_kernels);""", 1)

sub("""    const char *wroute = wmode == cv_triton_warp::Mode::Nearest""",
"""    if (dev.arch == 936 && req.src.type == ElemType::F32 && n.channels == 4 &&
        n.border_mode_code == 1 && wmode == cv_triton_warp::Mode::Nearest &&
        std::abs(n.A[3]) >= 0.125 && req.src.ptr % 8 == 0 && req.dst.ptr % 8 == 0 &&
        req.src.step % 8 == 0 && req.dst.step % 8 == 0 &&
        n.dst_w >= 256 && n.dst_h >= 64 && static_cast<long long>(n.dst_w) * n.dst_h >= 65536)
    {
        wkernel = "_warp_nearest_packed2_f32";
        n.block_h = 1;
        n.block_w = 256;
        n.num_warps = 4;
    }
    const char *wroute = wmode == cv_triton_warp::Mode::Nearest""", 1)

# Reuse validated aligned packing on gfx1100 and retain the generic
# linear warp dispatcher for transforms with small cross-row motion.
sub('    const bool cudaPacked = dev.arch == 936 && (spec.code == 0 || spec.code == 4) && !n.u16 && !n.f32 && n.n >= 65536 && n.cols % 4 == 0 &&\n',
    '    const bool cudaPacked = (dev.arch == 936 || dev.arch == 1100) && (spec.code == 0 || spec.code == 4) && !n.u16 && !n.f32 && n.n >= 65536 && n.cols % 4 == 0 &&\n', 1)
sub('    const bool packed = dev.arch == 936 && !n.u16 && !n.f32 && n.cols >= 1024 && n.cols % 4 == 0 &&\n',
    '    const bool packed = (dev.arch == 936 || dev.arch == 1100) && !n.u16 && !n.f32 && n.cols >= 1024 && n.cols % 4 == 0 &&\n', 1)
sub('    if (dev.arch == 936 && spec.code == 10 && !n.u16 && !n.f32 && n.n >= 65536 && n.cols % 4 == 0 &&\n',
    '    if ((dev.arch == 936 || dev.arch == 1100) && spec.code == 10 && !n.u16 && !n.f32 && n.n >= 65536 && n.cols % 4 == 0 &&\n', 1)
sub('    const bool packed4 = dev.arch == 936 && n.dst_h >= 64 && n.dst_w >= 256 && static_cast<long long>(n.dst_h) * n.dst_w >= 65536 && n.u8 && n.channels == 4 &&\n',
    '    // gfx1100 needs cross-row source reuse to benefit from linear packing.\n    const bool gfx1100LinearFallback = dev.arch == 1100 &&\n        n.mode == cv_triton_warp::Mode::Linear && std::abs(n.A[3]) < 0.125;\n    const bool packed4 = (dev.arch == 936 || dev.arch == 1100) && !gfx1100LinearFallback && n.dst_h >= 64 && n.dst_w >= 256 && static_cast<long long>(n.dst_h) * n.dst_w >= 65536 && n.u8 && n.channels == 4 &&\n', 1)
sub('    if (dev.arch == 936 && n.u8 && n.channels == 4 && n.border_mode_code == 1 &&\n',
    '    if ((dev.arch == 936 || dev.arch == 1100) && n.u8 && n.channels == 4 && n.border_mode_code == 1 &&\n', 1)
sub('    if (dev.arch == 936 && wmode == cv_triton_warp::Mode::Cubic && n.border_mode_code == 1 &&\n',
    '    if ((dev.arch == 936 || (dev.arch == 1100 && std::strcmp(wkernel, "_warp_cubic_packed4_u8") == 0)) &&\n        wmode == cv_triton_warp::Mode::Cubic && n.border_mode_code == 1 &&\n', 1)
sub('    if (dev.arch == 936 && n.u16 && n.channels == 4 && n.border_mode_code == 1 &&\n        wmode == cv_triton_warp::Mode::Nearest &&\n',
    '    if ((dev.arch == 936 || dev.arch == 1100) && n.u16 && n.channels == 4 && n.border_mode_code == 1 &&\n        wmode == cv_triton_warp::Mode::Nearest &&\n', 1)
sub('    if (dev.arch == 936 && req.src.type == ElemType::F32 && n.channels == 4 &&\n',
    '    if ((dev.arch == 936 || dev.arch == 1100) && req.src.type == ElemType::F32 && n.channels == 4 &&\n', 1)

dst = HERE / "src/runtime/hip_runtime.cpp"
if "--check" in sys.argv[1:]:
    if not dst.is_file() or dst.read_text() != out:
        print("src/runtime/hip_runtime.cpp is stale or missing; run tools/port_hip_runtime.py", file=sys.stderr)
        sys.exit(1)
    print("src/runtime/hip_runtime.cpp is current")
    sys.exit(0)
dst.write_text(out)
print(f"wrote src/runtime/hip_runtime.cpp ({out.count(chr(10))} lines)")
