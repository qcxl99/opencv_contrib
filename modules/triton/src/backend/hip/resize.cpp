// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// HIP backend: artifact selection, module load/cache, launch, error mapping.
// Sibling of backend/cuda/resize.cpp. No hand-written pixel arithmetic lives
// here -- the computation is entirely in the Triton-generated kernel.
//
// Differs from the CUDA backend in two load-bearing ways, not just API names:
//   - loads .hsaco (AMDGCN code object), not .cubin
//   - threads-per-block = num_warps * warp_size, with warp_size a compiled-in
//     constant for this artifact set (64, matching gfx936/BW1000 and recorded
//     in kernels_manifest.json by compile_resize_hip.py), not literal 32.
//     Same pattern as the CUDA backend's block_h/block_w/num_warps: the
//     manifest is the audit trail, this constant is what actually launches,
//     and the two must be kept in sync by hand if either changes.

#include "../../precomp.hpp"

#ifdef HAVE_TRITON_BACKEND

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include <hip/hip_runtime.h>

namespace cv { namespace triton { namespace backend {

namespace {

struct Artifact
{
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    // Compiled-in constants, kept in sync by hand with compile_resize_hip.py
    // (BLOCK_H/BLOCK_W/NUM_WARPS) and its manifest's recorded warp_size.
    int block_h = 1;
    int block_w = 256;
    int num_warps = 4;
    int warp_size = 64;   // gfx936; NOT read from the manifest at load time
    int shared_bytes = 0;
};

// Keyed by (context, variant) so a module is loaded once per context and
// reused. Loading is first-use only and therefore outside steady-state timing.
std::map<std::pair<hipCtx_t, std::string>, Artifact> g_cache;
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
 * NOTE (left as-is, HIP/BW1000 is out of scope for the CUBIC/C4/fx-fy/
 * float32 work that added the CUDA backend's dtype dimension): this key
 * format ("<mode>_cN_sp.dp.ss.ds.") no longer matches the CUDA backend's
 * ("<mode>_<dtype>_cN_sp.dp.ss.ds."), and kernels_hip.py/compile_resize_hip.py
 * still only emit LINEAR/NEAREST over uint8, 1/3 channels -- CUBIC, C4 and
 * float32 are CUDA-only right now. If this backend is ever extended to
 * match, the stride-unit bug fixed in backend/cuda/resize.cpp's
 * launchResize() (step is a BYTE pitch; the kernel's own stride argument is
 * an ELEMENT pitch, and the two only coincide for uint8) applies here
 * identically and would need the same CV_ELEM_SIZE1-based fix before any
 * non-uint8 dtype could work. */
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
    // request a small dynamic allocation. Same rule as the CUDA backend.
    return (variant.find("ds1") != std::string::npos) ? 256 : 0;
}

bool loadArtifact(hipCtx_t ctx, const std::string& variant, Artifact& out)
{
    const std::string path = std::string(artifactDir()) + "/" + variant + ".hsaco";
    const char* fn = (variant.rfind("linear", 0) == 0)
                         ? "_resize_linear_kernel" : "_resize_nearest_kernel";

    hipError_t r = hipModuleLoad(&out.module, path.c_str());
    if (r != hipSuccess)
    {
        std::snprintf(g_error, sizeof(g_error), "hipModuleLoad(%s) failed: %s",
                      path.c_str(), hipGetErrorString(r));
        return false;
    }
    r = hipModuleGetFunction(&out.function, out.module, fn);
    if (r != hipSuccess)
    {
        hipModuleUnload(out.module);
        out.module = nullptr;
        std::snprintf(g_error, sizeof(g_error), "hipModuleGetFunction(%s) failed: %s",
                      fn, hipGetErrorString(r));
        return false;
    }
    out.shared_bytes = sharedBytesFor(variant);
    if (out.shared_bytes > 0)
        hipFuncSetAttribute(reinterpret_cast<const void*>(out.function),
                            hipFuncAttributeMaxDynamicSharedMemorySize,
                            out.shared_bytes);
    (void)ctx;
    return true;
}

int divUp(int a, int b) { return (a + b - 1) / b; }

}  // namespace

const char* name() { return "hip-triton-aot"; }

bool available(int device)
{
    // The artifact directory must exist and hold at least one variant.
    const std::string probe = std::string(artifactDir()) + "/linear_c1_sp1dp1ss1ds1.hsaco";
    FILE* f = std::fopen(probe.c_str(), "rb");
    if (f == nullptr)
        return false;
    std::fclose(f);

    // Context readiness WITHOUT allocating or submitting work. Same reasoning
    // as the CUDA backend: a cold process has no current context, so
    // hipCtxGetCurrent() alone reported false for a perfectly usable backend.
    // The driver's primary context for the device answers the real question.
    if (hipInit(0) != hipSuccess)
        return false;

    hipCtx_t current = nullptr;
    if (hipCtxGetCurrent(&current) == hipSuccess && current != nullptr)
        return true;   // caller already has a context; nothing to probe

    hipDevice_t dev = 0;
    if (hipDeviceGet(&dev, device) != hipSuccess)
        return false;
    hipCtx_t primary = nullptr;
    if (hipDevicePrimaryCtxRetain(&primary, dev) != hipSuccess)
        return false;
    hipDevicePrimaryCtxRelease(dev);
    return primary != nullptr;
}

LaunchResult launchResize(const Context& c, const ResizeRequest& req)
{
    LaunchResult res;

    hipCtx_t hipctx = nullptr;
    if (hipCtxGetCurrent(&hipctx) != hipSuccess || hipctx == nullptr)
    {
        res.error_code = -1;
        res.message = "no current HIP context";
        return res;
    }

    const std::string variant = variantName(req);
    std::snprintf(res.variant, sizeof(res.variant), "%s", variant.c_str());

    Artifact art;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto key = std::make_pair(hipctx, variant);
        auto it = g_cache.find(key);
        if (it == g_cache.end())
        {
            Artifact fresh;
            if (!loadArtifact(hipctx, variant, fresh))
            {
                res.error_code = -2;
                res.message = g_error;
                return res;
            }
            it = g_cache.emplace(key, fresh).first;
        }
        art = it->second;
    }

    // Parameter ABI, read from the generated artifact (do not reorder):
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

    // threads-per-block = num_warps * warp_size, using this backend's
    // compiled-in warp_size (64) rather than the CUDA backend's 32.
    const hipError_t r = hipModuleLaunchKernel(
        art.function,
        static_cast<unsigned>(divUp(dst_h, art.block_h)),
        static_cast<unsigned>(divUp(dst_w, art.block_w)), 1,
        static_cast<unsigned>(art.num_warps * art.warp_size), 1, 1,
        static_cast<unsigned>(art.shared_bytes),
        reinterpret_cast<hipStream_t>(c.stream),
        params, nullptr);
    if (r != hipSuccess)
    {
        std::snprintf(g_error, sizeof(g_error), "hipModuleLaunchKernel: %s",
                      hipGetErrorString(r));
        res.error_code = -3;
        res.message = g_error;
        return res;
    }
    res.submitted = true;
    return res;
}

}}}  // namespace cv::triton::backend

#endif  // HAVE_TRITON_BACKEND
