// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

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
 *
 * dtype is ALSO a compile-time constant on the kernel side (OUT_FLOAT32:
 * tl.constexpr -- see kernels/resize.py): a float32 and a uint8 call with
 * otherwise-identical channels/alignment would collide on the same variant
 * string without this, silently loading the uint8 cubin for a float32
 * buffer (wrong element size, wrong result, no error). "u8"/"f32" is
 * spelled out rather than a bit, matching the alignment fields' own
 * self-describing style and keeping manifest filenames greppable.
 */
std::string variantName(const ResizeRequest& req)
{
    const char* mode = (req.interpolation == INTER_CUBIC) ? "cubic"
                      : (req.interpolation == INTER_LINEAR) ? "linear" : "nearest";
    const char* dtype = (req.src.depth() == CV_32F) ? "f32" : "u8";
    // The stride divisibility bit must be computed on the SAME unit the
    // kernel argument actually carries -- element count, not bytes (see
    // launchResize's src_step/dst_step derivation and its comment). The
    // pointer bit stays byte-based: tt.divisibility on a tl.pointer is a
    // byte-address property regardless of element dtype, and req.*.ptr
    // already IS a byte address. Byte stride and element stride coincide
    // for uint8 (1 byte/element), which is why this split was invisible
    // before float32 existed: a byte-stride-divisible-by-16 buffer is not
    // always element-stride-divisible-by-16 once elements are 4 bytes.
    const int elem_size = CV_ELEM_SIZE1(req.src.type);  // dst shares src's depth (checked in resizeCore)
    const long src_stride_elems = static_cast<long>(req.src.step) / elem_size;
    const long dst_stride_elems = static_cast<long>(req.dst.step) / elem_size;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s_%s_c%d_sp%ddp%dss%dds%d",
                  mode, dtype, req.src.channels(),
                  (req.src.ptr % 16 == 0) ? 1 : 0,
                  (req.dst.ptr % 16 == 0) ? 1 : 0,
                  (src_stride_elems % 16 == 0) ? 1 : 0,
                  (dst_stride_elems % 16 == 0) ? 1 : 0);
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
    // variant is "<mode>_<dtype>_c...": mode is the first '_'-delimited
    // field, same prefix test as before just re-anchored past it. NEAREST
    // has no OUT_FLOAT32 specialization (see kernels/resize.py:
    // _resize_nearest_kernel's load/store is a plain byte copy with no
    // round/saturate step, so it is dtype-transparent by construction) --
    // one function name serves both nearest_u8 and nearest_f32 variants.
    const char* fn = (variant.rfind("linear_", 0) == 0) ? "_resize_linear_kernel"
                    : (variant.rfind("cubic_", 0) == 0)  ? "_resize_cubic_kernel"
                                                          : "_resize_nearest_kernel";

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
    const std::string probe = std::string(artifactDir()) + "/linear_u8_c1_sp1dp1ss1ds1.cubin";
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
    // Shared verbatim by linear, nearest AND cubic -- confirmed against
    // kernels/resize.py's _resize_cubic_kernel signature, which takes the
    // same (src_ptr, dst_ptr, src_h, src_w, dst_h, dst_w, src_stride_h,
    // dst_stride_h, channels, fx, fy, BLOCK_H, BLOCK_W) shape as linear and
    // nearest. dtype does not change this ABI either: OUT_FLOAT32 is a
    // tl.constexpr baked into the cubin at compile time, not a runtime
    // argument, so the f32 and u8 variants of the same mode launch with
    // identical params[].
    int src_h = req.src.rows, src_w = req.src.cols;
    int dst_h = req.dst.rows, dst_w = req.dst.cols;
    // src_stride_h/dst_stride_h are ELEMENT strides on the kernel side
    // (kernels/resize.py addresses as `row * stride_h + col * channels`,
    // one element per index step, for BOTH uint8 and float32 -- the
    // kernel source has no unit conversion anywhere). req.src.step /
    // req.dst.step are BYTE pitches (see backend::ImageView's own doc
    // comment: "row pitch IN BYTES"). Those two units coincide for uint8
    // (1 byte/element) and silently diverge by elemSize() for float32 (4
    // bytes/element) -- confirmed by direct cuLaunchKernel reproduction:
    // passing the byte stride for a float32 NEAREST call corrupts the
    // output (wrong rows read/written, matching a stride 4x too wide).
    // Dividing by elemSize1() (bytes per CHANNEL, not per pixel -- step is
    // already a whole-row byte count, and the kernel's own stride unit is
    // "elements of one channel", matching how it indexes `col * channels`
    // separately) converts byte pitch to the element pitch the kernel
    // actually expects.
    const int src_elem_size = CV_ELEM_SIZE1(req.src.type);
    const int dst_elem_size = CV_ELEM_SIZE1(req.dst.type);
    int src_step = static_cast<int>(req.src.step) / src_elem_size;
    int dst_step = static_cast<int>(req.dst.step) / dst_elem_size;
    // req.fx/fy, NOT src_w/dst_w recomputed here: resizeCore() already
    // resolved the right value for both the dsize-given case (src/dsize,
    // which IS src_w/dst_w) and the fx/fy-given case (1/fx, 1/fy -- NOT
    // re-derived from the rounded dsize). Recomputing from the integer
    // dims here would silently collapse that distinction and disagree
    // with cv::cuda::resize on every non-exact scale factor -- see
    // opencv2/triton/core.hpp's resizeCore() doc comment.
    float fx = static_cast<float>(req.fx);
    float fy = static_cast<float>(req.fy);
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
