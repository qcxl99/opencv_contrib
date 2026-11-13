#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

"""Derive src/runtime/hip_runtime.cpp from src/runtime/cuda_runtime.cpp.

    port_hip_runtime.py            write src/runtime/hip_runtime.cpp
    port_hip_runtime.py --check    exit 1 if the checked-in file is not what this would write

The CUDA file is the source: the HIP runtime is the same code with the driver API renamed and
the few calls whose signature differs rewritten (allocation into void*, the 2-D copy, the module
load and launch, the per-device target query). Every substitution is an exact-count
replacement, so a CUDA edit this script does not know fails the port instead of leaving CUDA
code -- or a silently different HIP runtime -- in the generated file.
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
'''// Runtime JIT, HIP driver side (Hygon BW1000 / gfx936, DTK). See runtime.hpp for execution and ownership rules.
//
// GENERATED from cuda_runtime.cpp by tools/port_hip_runtime.py -- do not edit; edit the CUDA
// file and re-run the port. Exactly one of the two is compiled into a build (CMakeLists.txt).
//
// What differs from the CUDA file, and why -- everything else is the same code:
//   * Target: backend "hip"; arch = the gfx number parsed from gcnArchName ("gfx936:sramecc+:..."
//     -> 936; an arch with a letter suffix such as gfx90a has no integer form in the key and is
//     refused); warp_size from the device (64 on gfx936, not 32).
//   * Kernels: kernels/<op>_cpu_hip.py where the CUDA source calls a libdevice primitive
//     the HIP compiler lacks; tools/make_kernels_cpu_hip.py derives resize, cvtColor
//     and warpAffine sources and rejects stale copies.
//   * Binary: manifest format "hsaco", loaded with hipModuleLoadData.
//   * Compile helper environment: ROCM_PATH is forwarded (the hip compiler finds clang and its
//     device libraries through it), and the GPU is hidden from the helper with the HIP and ROCR
//     visibility variables as CUDA_VISIBLE_DEVICES= does for cuda.
//   * Dynamic shared memory above the 64 KiB LDS is refused rather than requested through an
//     attribute call, which HIP defines for host function pointers, not module functions.
//   * The hipCtx* family is marked deprecated in HIP. It is what DTK offers for explicit context
//     handling and what this runtime's context identity needs; the warning is silenced here only.
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
# Float cvtColor now uses explicit libdevice rounding too; derive its HIP source.
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
out = out.replace('if (base) { if (stream_ordered) cuMemFreeAsync(base, nullptr); else cuMemFree(base); }',
                  'if (base)\n            {\n                void* p = reinterpret_cast<void*>(base);\n'
                  '                if (stream_ordered) (void)hipFreeAsync(p, nullptr); else (void)hipFree(p);\n            }')
alloc_old = '''    CUresult r = t->stream_ordered ? cuMemAllocAsync(&t->base, t->bytes, stream) : cuMemAlloc(&t->base, t->bytes);
    if (r == CUDA_SUCCESS && poison) r = cuMemsetD8Async(t->base, 0xFF, t->bytes, stream);
    if (r == CUDA_SUCCESS) r = cuMemcpyHtoDAsync(t->base, t->host.data(), t->bytes, stream);
    if (r == CUDA_SUCCESS) r = cuEventCreate(&t->uploaded, CU_EVENT_DISABLE_TIMING);
    if (r == CUDA_SUCCESS) r = cuEventRecord(t->uploaded, stream);'''
alloc_new = '''    void* dptr = nullptr;
    hipError_t r = t->stream_ordered ? hipMallocAsync(&dptr, t->bytes, stream) : hipMalloc(&dptr, t->bytes);
    t->base = reinterpret_cast<std::uintptr_t>(dptr);
    if (r == hipSuccess && poison) r = hipMemsetD8Async(dptr, 0xFF, t->bytes, stream);
    if (r == hipSuccess) r = hipMemcpyHtoDAsync(dptr, t->host.data(), t->bytes, stream);
    if (r == hipSuccess) r = hipEventCreateWithFlags(&t->uploaded, hipEventDisableTiming);
    if (r == hipSuccess) r = hipEventRecord(t->uploaded, stream);'''
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
    int arch = 0;                // 936 for gfx936
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

# ---- 6. plain token renames -----------------------------------------------------------
pairs = [
    ("CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED", "hipErrorStreamCaptureUnsupported"),
    ("CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED", "hipDeviceAttributeMemoryPoolsSupported"),
    ("CU_EVENT_DISABLE_TIMING", "hipEventDisableTiming"),
    ("CUDA_ERROR_NOT_READY", "hipErrorNotReady"),
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

dst = HERE / "src/runtime/hip_runtime.cpp"
if "--check" in sys.argv[1:]:
    if not dst.is_file() or dst.read_text() != out:
        print("src/runtime/hip_runtime.cpp is stale or missing; run tools/port_hip_runtime.py", file=sys.stderr)
        sys.exit(1)
    print("src/runtime/hip_runtime.cpp is current")
    sys.exit(0)
dst.write_text(out)
print(f"wrote src/runtime/hip_runtime.cpp ({out.count(chr(10))} lines)")
