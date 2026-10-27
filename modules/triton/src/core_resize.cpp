// Platform-agnostic core: every cv::triton::resize validation rule, applied
// directly to ImageView/ExecutionContext. No cuda::GpuMat, no cuda::Stream,
// no CUDA/HIP header. The NVIDIA GpuMat adapter (resize.cpp) and any other
// adapter (e.g. a BW1000 torch-tensor-backed caller) both reduce to a call
// into resizeCore() after unwrapping their own buffer type into the two POD
// structs below.

#include "opencv2/triton/core.hpp"
#include "opencv2/imgproc.hpp"   // INTER_LINEAR / INTER_NEAREST
#include "backend/backend.hpp"

namespace cv { namespace triton {

// NOTE: haveBackend()/backendName() are NOT here. Both stay adapter-side
// (resize.cpp for the GpuMat/CUDA adapter): their "current device" is a
// CUDA-specific global query (cuda::getDevice()), which has no
// platform-agnostic equivalent at this layer, and adding one would either
// invent a fake default device or silently change the existing NVIDIA
// semantics. A BW1000 (or other) adapter that knows its own device id can
// call backend::available(device) directly without needing a core-layer
// wrapper for it.

namespace detail {

/** ImageView (public, stable) -> backend::ImageView (module-private). Same
 *  shape; kept as two structs so the public header never pulls in the
 *  backend-private one, and so backend-internal fields can change without
 *  touching the public ABI. */
static backend::ImageView toBackendView(const ImageView& v)
{
    backend::ImageView b;
    b.ptr = v.ptr;
    b.rows = v.rows;
    b.cols = v.cols;
    b.type = v.type;
    b.step = v.step;
    b.whole_rows = v.whole_rows;
    b.whole_cols = v.whole_cols;
    b.offset_y = v.offset_y;
    b.offset_x = v.offset_x;
    return b;
}

}  // namespace detail

void resizeCore(const ImageView& src, const ImageView& dst, Size dsize,
                int interpolation, const ExecutionContext& ctx)
{
    // --- supported subset, enforced rather than coerced -------------------
    if (src.rows <= 0 || src.cols <= 0)
        CV_Error(Error::StsBadArg, "cv::triton::resizeCore: src must be non-empty");
    if (src.depth() != CV_8U)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::resizeCore: only CV_8U is supported in this round");
    if (src.channels() != 1 && src.channels() != 3)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::resizeCore: only 1 or 3 channels are supported in this round");
    if (interpolation != INTER_NEAREST && interpolation != INTER_LINEAR)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::resizeCore: only INTER_NEAREST and INTER_LINEAR are supported");
    if (dsize.width <= 0 || dsize.height <= 0)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::resizeCore: dsize must be explicit and non-empty");
    if (dst.rows != dsize.height || dst.cols != dsize.width)
        CV_Error(Error::StsBadArg,
                 "cv::triton::resizeCore: dst must already be sized to dsize; "
                 "this layer does not allocate");
    if (dst.type != src.type)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::resizeCore: dst type must match src type");

    // ROI src/dst are supported: both backends already address by the real
    // device row pitch (step) read out of the view, which is exactly the
    // same step a ROI GpuMat reports -- there is no separate "whole buffer"
    // addressing path to fall back to incorrectly. See the upstream decision
    // in opencv_contrib@b8a2073 (modules/triton: sync GpuMat layer with
    // kernel P0/P1), which removed the equivalent isRoiView() gates from the
    // CUDA adapter for the same reason.

    if (dst.ptr == src.ptr)
        CV_Error(Error::StsBadArg, "cv::triton::resizeCore: in-place resize is not supported");
    {
        const std::uintptr_t s0 = src.ptr;
        const std::uintptr_t s1 = s0 + src.step * static_cast<std::size_t>(src.rows);
        const std::uintptr_t d0 = dst.ptr;
        const std::uintptr_t d1 = d0 + dst.step * static_cast<std::size_t>(dst.rows);
        if (d0 < s1 && s0 < d1)
            CV_Error(Error::StsBadArg, "cv::triton::resizeCore: src and dst overlap");
    }

#ifdef HAVE_TRITON_BACKEND
    const bool backendUsable = backend::available(ctx.device);
#else
    const bool backendUsable = false;
#endif
    if (!backendUsable)
        CV_Error(Error::StsNotImplemented,
                 "cv::triton::resizeCore: no usable Triton backend for ctx.device. "
                 "This module never falls back to CPU or cv::cuda");

    backend::Context bctx;
    bctx.device = ctx.device;
    bctx.stream = ctx.stream;
    bctx.backend_name = backend::name();

    backend::ResizeRequest req;
    req.src = detail::toBackendView(src);
    req.dst = detail::toBackendView(dst);
    req.dsize = dsize;
    req.interpolation = interpolation;
    req.fx = static_cast<double>(dsize.width) / src.cols;
    req.fy = static_cast<double>(dsize.height) / src.rows;

    const backend::LaunchResult r = backend::launchResize(bctx, req);
    if (!r.submitted)
        CV_Error(Error::StsError,
                 cv::format("cv::triton::resizeCore: launch failed (%d): %s",
                            r.error_code, r.message));
    // Asynchronous: everything needed is on ctx.stream. No implicit sync here.
}

}}  // namespace cv::triton
