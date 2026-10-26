// CUDA backend: artifact selection, module load/cache, launch, error mapping.
// No hand-written pixel arithmetic lives here -- the computation is entirely
// in the Triton-generated kernel.

#include "../../precomp.hpp"

#ifdef HAVE_TRITON_BACKEND

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include <cuda.h>

namespace cv { namespace triton { namespace backend {

namespace {

struct Artifact
{
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    int block_h = 1;
    int block_w = 256;
    int num_warps = 4;
    int shared_bytes = 0;
};

// Keyed by (context, variant) so a module is loaded once per context and
// reused. Loading is first-use only and therefore outside steady-state timing.
std::map<std::pair<CUcontext, std::string>, Artifact> g_cache;
std::mutex g_mutex;
char g_error[256] = {0};

const char* artifactDir()
{
    // Build-time location of the generated artifacts, overridable for testing.
    const char* d = std::getenv("OPENCV_TRITON_ARTIFACTS");
    return (d != nullptr) ? d : OPENCV_TRITON_ARTIFACT_DIR;
}

/** Variant name from the runtime signature.
 *
 * Triton specializes on divisibility-by-16 of BOTH the pointer and the byte
 * stride, and `channels` is a compile-time constant, so all four must be in
 * the key. Keying on stride alone produced misaligned accesses during
 * development: an offset-but-contiguous buffer selected a cubin compiled under
 * an alignment assumption it did not satisfy.
 */
std::string variantName(const ResizeRequest& req)
{
    const char* mode = (req.interpolation == INTER_LINEAR) ? "linear" : "nearest";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s_c%d_sp%ddp%dss%dds%d",
                  mode, req.src.channels(),
                  (req.src.ptr % 16 == 0) ? 1 : 0,
                  (req.dst.ptr % 16 == 0) ? 1 : 0,
                  (req.src.step % 16 == 0) ? 1 : 0,
                  (req.dst.step % 16 == 0) ? 1 : 0);
    return std::string(buf);
}

int sharedBytesFor(const std::string& variant)
{
    // Recorded per variant in the manifest; the aligned-destination variants
    // request a small dynamic allocation.
    return (variant.find("ds1") != std::string::npos) ? 256 : 0;
}

bool loadArtifact(CUcontext ctx, const std::string& variant, Artifact& out)
{
    const std::string path = std::string(artifactDir()) + "/" + variant + ".cubin";
    const char* fn = (variant.rfind("linear", 0) == 0)
                         ? "_resize_linear_kernel" : "_resize_nearest_kernel";

    CUresult r = cuModuleLoad(&out.module, path.c_str());
    if (r != CUDA_SUCCESS)
    {
        std::snprintf(g_error, sizeof(g_error), "cuModuleLoad(%s) failed", path.c_str());
        return false;
    }
    r = cuModuleGetFunction(&out.function, out.module, fn);
    if (r != CUDA_SUCCESS)
    {
        cuModuleUnload(out.module);
        out.module = nullptr;
        std::snprintf(g_error, sizeof(g_error), "cuModuleGetFunction(%s) failed", fn);
        return false;
    }
    out.shared_bytes = sharedBytesFor(variant);
    if (out.shared_bytes > 0)
        cuFuncSetAttribute(out.function,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           out.shared_bytes);
    (void)ctx;
    return true;
}

int divUp(int a, int b) { return (a + b - 1) / b; }

}  // namespace

const char* name() { return "cuda-triton-aot"; }

bool available(int device)
{
    // The artifact directory must exist and hold at least one variant.
    const std::string probe = std::string(artifactDir()) + "/linear_c1_sp1dp1ss1ds1.cubin";
    FILE* f = std::fopen(probe.c_str(), "rb");
    if (f == nullptr)
        return false;
    std::fclose(f);

    // Context readiness WITHOUT allocating or submitting work. A cold process
    // has no current context, so cuCtxGetCurrent() alone reported false for a
    // perfectly usable backend. The driver's primary context for the device
    // answers the real question -- can this device run our artifacts -- and
    // retain/release is reference-counted, not an allocation.
    if (cuInit(0) != CUDA_SUCCESS)
        return false;

    CUcontext current = nullptr;
    if (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current != nullptr)
        return true;   // caller already has a context; nothing to probe

    CUdevice dev = 0;
    if (cuDeviceGet(&dev, device) != CUDA_SUCCESS)
        return false;
    CUcontext primary = nullptr;
    if (cuDevicePrimaryCtxRetain(&primary, dev) != CUDA_SUCCESS)
        return false;
    cuDevicePrimaryCtxRelease(dev);
    return primary != nullptr;
}

LaunchResult launchResize(const Context& c, const ResizeRequest& req)
{
    LaunchResult res;

    CUcontext cuctx = nullptr;
    if (cuCtxGetCurrent(&cuctx) != CUDA_SUCCESS || cuctx == nullptr)
    {
        res.error_code = -1;
        res.message = "no current CUDA context";
        return res;
    }

    const std::string variant = variantName(req);
    std::snprintf(res.variant, sizeof(res.variant), "%s", variant.c_str());

    Artifact art;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto key = std::make_pair(cuctx, variant);
        auto it = g_cache.find(key);
        if (it == g_cache.end())
        {
            Artifact fresh;
            if (!loadArtifact(cuctx, variant, fresh))
            {
                res.error_code = -2;
                res.message = g_error;
                return res;
            }
            it = g_cache.emplace(key, fresh).first;
        }
        art = it->second;
    }

    // Parameter ABI, read from the generated PTX (do not reorder):
    //   ptr, ptr, u32 x6, f32 x2, ptr, ptr(scratch)
    int src_h = req.src.rows, src_w = req.src.cols;
    int dst_h = req.dst.rows, dst_w = req.dst.cols;
    int src_step = static_cast<int>(req.src.step);
    int dst_step = static_cast<int>(req.dst.step);
    float fx = static_cast<float>(src_w) / static_cast<float>(dst_w);
    float fy = static_cast<float>(src_h) / static_cast<float>(dst_h);
    void* src_ptr = reinterpret_cast<void*>(req.src.ptr);
    void* dst_ptr = reinterpret_cast<void*>(req.dst.ptr);
    void* scratch = nullptr;

    void* params[] = { &src_ptr, &dst_ptr,
                       &src_h, &src_w, &dst_h, &dst_w,
                       &src_step, &dst_step,
                       &fx, &fy,
                       &scratch, &scratch };

    const CUresult r = cuLaunchKernel(
        art.function,
        static_cast<unsigned>(divUp(dst_h, art.block_h)),
        static_cast<unsigned>(divUp(dst_w, art.block_w)), 1,
        static_cast<unsigned>(art.num_warps * 32), 1, 1,
        static_cast<unsigned>(art.shared_bytes),
        reinterpret_cast<CUstream>(c.stream),
        params, nullptr);
    if (r != CUDA_SUCCESS)
    {
        const char* nm = nullptr;
        cuGetErrorName(r, &nm);
        std::snprintf(g_error, sizeof(g_error), "cuLaunchKernel: %s", nm ? nm : "unknown");
        res.error_code = -3;
        res.message = g_error;
        return res;
    }
    res.submitted = true;
    return res;
}

}}}  // namespace cv::triton::backend

#endif  // HAVE_TRITON_BACKEND
