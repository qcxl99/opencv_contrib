#include "precomp.hpp"

namespace cv { namespace triton {

bool haveBackend()
{
#ifdef HAVE_TRITON_BACKEND
    if (cuda::getCudaEnabledDeviceCount() < 1)
        return false;
    // Pure query: no device memory is allocated and no work is submitted.
    //
    // It previously allocated a 1x1 GpuMat and called setTo purely to force a
    // CUDA context into existence, because cuCtxGetCurrent() returns null on a
    // cold process and the probe then reported false for a working backend
    // (the test suite silently skipped all of its cases). Side-effecting
    // "have" predicates are the wrong fix: no other cv::cuda:: predicate
    // allocates. Context readiness is now the CUDA backend's concern, where
    // the driver API it needs already lives.
    return backend::available(cuda::getDevice());
#else
    return false;
#endif
}

String backendName()
{
#ifdef HAVE_TRITON_BACKEND
    return haveBackend() ? String(backend::name()) : String();
#else
    return String();
#endif
}

#ifndef HAVE_TRITON_BACKEND

void resize(InputArray, OutputArray, Size, double, double, int, cuda::Stream&)
{
    CV_Error(Error::StsNotImplemented,
             "cv::triton::resize requires the module to be built with a Triton backend "
             "(-DWITH_TRITON=ON and a supported TRITON_BACKEND_*)");
}

#else

/** Is this GpuMat a view into a larger allocation? */
static bool isRoiView(const cuda::GpuMat& m)
{
    Size whole;
    Point ofs;
    m.locateROI(whole, ofs);
    return ofs.x != 0 || ofs.y != 0 || whole != m.size();
}

static backend::ImageView viewOf(const cuda::GpuMat& m)
{
    Size whole;
    Point ofs;
    m.locateROI(whole, ofs);

    backend::ImageView v;
    v.ptr = reinterpret_cast<std::uintptr_t>(m.cudaPtr());
    v.rows = m.rows;
    v.cols = m.cols;
    v.type = m.type();
    v.step = m.step;
    v.whole_rows = whole.height;
    v.whole_cols = whole.width;
    v.offset_y = ofs.y;
    v.offset_x = ofs.x;
    return v;
}

void resize(InputArray _src, OutputArray _dst, Size dsize,
            double fx, double fy, int interpolation, cuda::Stream& stream)
{
    CV_Assert(_src.kind() == _InputArray::CUDA_GPU_MAT);
    cuda::GpuMat src = _src.getGpuMat();

    // --- supported subset, enforced rather than coerced -------------------
    CV_Assert(!src.empty());
    if (src.depth() != CV_8U)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::resize: only CV_8U is supported in this round");
    if (src.channels() != 1 && src.channels() != 3)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::resize: only 1 or 3 channels are supported in this round");
    if (interpolation != INTER_NEAREST && interpolation != INTER_LINEAR)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::resize: only INTER_NEAREST and INTER_LINEAR are supported");

    // Explicit dsize only. fx/fy derive dsize through a different rounding
    // rule (saturate_cast<int>(src*scale)); accepting them silently would
    // change results, so they are rejected while unsupported.
    if (dsize.empty())
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::resize: explicit non-empty dsize is required; the fx/fy "
                 "scale-factor path is not supported in this round");
    if (fx != 0 || fy != 0)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::resize: fx/fy must be 0 while only explicit dsize is supported");

    // A ROI view has a different addressing contract; reject rather than
    // silently copying it to a contiguous buffer. Checked on BOTH sides: only
    // the source was checked before, so a ROI destination reached the kernel
    // with a pitch the artifact selection had not accounted for.
    if (isRoiView(src))
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::resize: ROI source views are not supported in this round");
    if (_dst.kind() == _InputArray::CUDA_GPU_MAT)
    {
        const cuda::GpuMat preDst = _dst.getGpuMat();
        if (!preDst.empty() && isRoiView(preDst))
            CV_Error(Error::StsUnsupportedFormat,
                     "cv::triton::resize: ROI destination views are not supported "
                     "in this round");
    }

    // Detect in-place BEFORE _dst.create(): create() may reallocate dst when
    // the requested size differs, after which dst.cudaPtr() no longer equals
    // src.cudaPtr() and the aliasing is invisible. Compare the caller's
    // arrays, not the post-create buffers.
    if (_dst.kind() == _InputArray::CUDA_GPU_MAT)
    {
        const cuda::GpuMat preDst = _dst.getGpuMat();
        if (!preDst.empty() && preDst.cudaPtr() == src.cudaPtr())
            CV_Error(Error::StsBadArg,
                     "cv::triton::resize: in-place resize is not supported");
    }

    if (!haveBackend())
        CV_Error(Error::StsNotImplemented,
                 "cv::triton::resize: no usable Triton backend for the current device. "
                 "This module never falls back to CPU or cv::cuda");

    // --- prepare the output, THEN read its address ------------------------
    // create() may reallocate, so the pointer, step and ownership must be
    // re-read afterwards rather than captured before.
    _dst.create(dsize, src.type());
    cuda::GpuMat dst = _dst.getGpuMat();
    CV_Assert(dst.rows == dsize.height && dst.cols == dsize.width);
    CV_Assert(dst.type() == src.type());
    // create() returns the caller's buffer unchanged when the size already
    // matches, so a ROI destination can still arrive here even though the
    // pre-create check passed on an empty dst.
    if (isRoiView(dst))
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::resize: ROI destination views are not supported "
                 "in this round");

    if (dst.cudaPtr() == src.cudaPtr())
        CV_Error(Error::StsBadArg, "cv::triton::resize: in-place resize is not supported");
    {
        const char* s0 = static_cast<const char*>(src.cudaPtr());
        const char* s1 = s0 + src.step * src.rows;
        const char* d0 = static_cast<const char*>(dst.cudaPtr());
        const char* d1 = d0 + dst.step * dst.rows;
        if (d0 < s1 && s0 < d1)
            CV_Error(Error::StsBadArg,
                     "cv::triton::resize: src and dst overlap");
    }

    backend::Context ctx;
    ctx.device = cuda::getDevice();
    ctx.stream = reinterpret_cast<std::uintptr_t>(cuda::StreamAccessor::getStream(stream));
    ctx.backend_name = backend::name();

    backend::ResizeRequest req;
    req.src = viewOf(src);
    req.dst = viewOf(dst);
    req.dsize = dsize;
    req.interpolation = interpolation;
    // Scale factors as the native implementation computes them from dsize.
    req.fx = static_cast<double>(dsize.width) / src.cols;
    req.fy = static_cast<double>(dsize.height) / src.rows;

    const backend::LaunchResult r = backend::launchResize(ctx, req);
    if (!r.submitted)
        CV_Error(Error::StsError,
                 cv::format("cv::triton::resize: launch failed (%d): %s",
                            r.error_code, r.message));
    // Asynchronous: everything needed is on `stream`. No implicit sync here.
}

#endif  // HAVE_TRITON_BACKEND

}}  // namespace cv::triton
