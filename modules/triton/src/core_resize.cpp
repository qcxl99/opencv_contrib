// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Platform-agnostic core of cv::triton::resize. Both entries -- resizeCore() (device pointers)
// and the GpuMat adapter (cuda_resize_adapter.cpp) -- reduce to detail::resizeDevice(), which
// hands the request to the runtime JIT (src/runtime/runtime.hpp) and maps its status to an
// OpenCV error. Numerics: OpenCV CPU cv::resize as the pinned reference configuration
// (kernelforge-lab contract resize/0.15.0), not cv::cuda::resize. There is no fallback to CPU or
// to cv::cuda: a call that returns has submitted the computation (or the identity copy).

#include "precomp.hpp"
#include "runtime/cvtcolor.hpp"
#include "runtime/warpaffine.hpp"
#include "runtime/runtime.hpp"

namespace cv { namespace triton {

namespace detail {

kf_resize_rt::DeviceView toDeviceView(const ImageView& v, int device, std::uintptr_t alloc_base, std::size_t alloc_bytes)
{
    kf_resize_rt::DeviceView d;
    d.ctx.device = device;            // the runtime identifies the context itself (cuCtxGetId)
    d.ptr = v.ptr;                    // the first element of the view: the ROI offset is already applied
    d.rows = v.rows;
    d.cols = v.cols;
    d.channels = CV_MAT_CN(v.type);
    const int depth = CV_MAT_DEPTH(v.type);
    if (depth != CV_8U && depth != CV_32F && depth != CV_16U)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton: only CV_8U, CV_16U and CV_32F are supported");
    d.type = depth == CV_8U ? kf_resize_rt::ElemType::U8
             : (depth == CV_16U ? kf_resize_rt::ElemType::U16 : kf_resize_rt::ElemType::F32);
    d.step = v.step;
    d.alloc_base = alloc_base;
    d.alloc_bytes = alloc_bytes;
    return d;
}

static void raiseCvt(const kf_resize_rt::Status& s)
{
    using kf_resize_rt::Reason;
    const std::string what = cv::format("cv::triton::cvtColor: %s (%s)", s.message.c_str(),
                                        kf_resize_rt::reason_name(s.reason));
    if (s.code == kf_resize_rt::Code::Unsupported)
    {
        switch (s.reason)
        {
        case Reason::DepthOrChannels: case Reason::Interpolation: case Reason::TypeMismatch:
        case Reason::RouteNotEnabled:
            CV_Error(Error::StsUnsupportedFormat, what);
        default:
            CV_Error(Error::StsBadArg, what);
        }
    }
    CV_Error(Error::StsError, what + (s.submitted ? " -- work had already been submitted; dst is undefined" : ""));
}

static void raise(const kf_resize_rt::Status& s)
{
    using kf_resize_rt::Reason;
    const std::string what = cv::format("cv::triton::resize: %s (%s)", s.message.c_str(), kf_resize_rt::reason_name(s.reason));
    if (s.code == kf_resize_rt::Code::Unsupported)
    {
        switch (s.reason)
        {
        case Reason::DepthOrChannels: case Reason::Interpolation: case Reason::TypeMismatch: case Reason::RouteNotEnabled:
            CV_Error(Error::StsUnsupportedFormat, what);
        default:
            CV_Error(Error::StsBadArg, what);
        }
    }
    CV_Error(Error::StsError, what + (s.submitted ? " -- work had already been submitted; dst is undefined" : ""));
}

void resizeDevice(const kf_resize_rt::DeviceView& src, const kf_resize_rt::DeviceView& dst, Size dsize,
                  double fx, double fy, int interpolation, int device, std::uintptr_t stream)
{
    kf_resize_rt::ResizeRequest req;
    req.src = src;
    req.dst = dst;
    req.dsize_w = dsize.width;
    req.dsize_h = dsize.height;
    req.fx = fx;
    req.fy = fy;
    req.interpolation = interpolation;
    kf_resize_rt::ExecContext ex;
    ex.ctx.device = device;
    ex.stream = stream;
    const kf_resize_rt::Status s = runtime::resize(req, ex);
    if (s.code != kf_resize_rt::Code::Submitted)
        raise(s);
    // Asynchronous: everything is on `stream`. No implicit synchronisation.
}

static void raiseWarp(const kf_resize_rt::Status& s)
{
    using kf_resize_rt::Reason;
    const std::string what = cv::format("cv::triton::warpAffine: %s (%s)", s.message.c_str(),
                                        kf_resize_rt::reason_name(s.reason));
    if (s.code == kf_resize_rt::Code::Unsupported)
    {
        switch (s.reason)
        {
        case Reason::DepthOrChannels: case Reason::Interpolation: case Reason::TypeMismatch:
        case Reason::RouteNotEnabled:
            CV_Error(Error::StsUnsupportedFormat, what);
        default:
            CV_Error(Error::StsBadArg, what);
        }
    }
    CV_Error(Error::StsError, what + (s.submitted ? " -- work had already been submitted; dst is undefined" : ""));
}

void warpAffineDevice(const kf_warpaffine::Request& req, int device, std::uintptr_t stream)
{
    kf_resize_rt::ExecContext ex;
    ex.ctx.device = device;
    ex.stream = stream;
    const kf_resize_rt::Status s = runtime::warpAffine(req, ex);
    if (s.code != kf_resize_rt::Code::Submitted)
        raiseWarp(s);
    // Asynchronous: same contract as resizeDevice().
}

void cvtColorDevice(const kf_cvtcolor::Request& req, int device, std::uintptr_t stream)
{
    kf_resize_rt::ExecContext ex;
    ex.ctx.device = device;
    ex.stream = stream;
    const kf_resize_rt::Status s = runtime::cvtColor(req, ex);
    if (s.code != kf_resize_rt::Code::Submitted)
        raiseCvt(s);
    // Asynchronous: same contract as resizeDevice().
}

}  // namespace detail

void resizeCore(const ImageView& src, const ImageView& dst, Size dsize,
                int interpolation, const ExecutionContext& ctx,
                double scaleX, double scaleY)
{
    if (dsize.width <= 0 || dsize.height <= 0)
        CV_Error(Error::StsUnsupportedFormat, "cv::triton::resizeCore: dsize must be explicit and non-empty");
    // resize covers CV_8U and CV_32F only; CV_16U belongs to warpAffine. Checked here because
    // toDeviceView is shared and now accepts all three depths -- refusing later would let the
    // request reach the compile helper, which reports -2 instead of the documented
    // StsUnsupportedFormat.
    for (const ImageView* v : {&src, &dst})
    {
        const int depth = CV_MAT_DEPTH(v->type);
        if (depth != CV_8U && depth != CV_32F)
            CV_Error(Error::StsUnsupportedFormat,
                     cv::format("cv::triton::resize: depth %d is not supported (CV_8U and CV_32F only)",
                                depth));
    }
    if (!runtime::deviceAvailable(ctx.device))
        CV_Error(Error::StsNotImplemented, "cv::triton::resizeCore: no usable CUDA device for ctx.device; "
                                           "this module never falls back to CPU or cv::cuda");
    // scaleX/scaleY are kernel-space (source pixels per destination pixel); the computation uses
    // the inverse, as cv::resize's fx/fy. 1/(1/fx) is not always fx in double, so a caller that
    // holds the original fx/fy and needs bit-exactness with cv::resize(fx, fy) should use the
    // GpuMat entry, which passes them through unchanged.
    const double fx = scaleX > 0 ? 1.0 / scaleX : 0.0;
    const double fy = scaleY > 0 ? 1.0 / scaleY : 0.0;
    // dsize is explicit here, so cv::resize's precedence ignores fx/fy unless both scales are
    // given: then they are the inverse scales and dsize must already equal cvRound(src * f).
    Size ds = dsize;
    if (fx > 0 && fy > 0)
    {
        if (dsize != Size(cvRound(src.cols * fx), cvRound(src.rows * fy)))
            CV_Error(Error::StsBadArg, "cv::triton::resizeCore: dsize disagrees with scaleX/scaleY");
        ds = Size();
    }
    detail::resizeDevice(detail::toDeviceView(src, ctx.device, 0, 0), detail::toDeviceView(dst, ctx.device, 0, 0),
                         ds, fx, fy, interpolation, ctx.device, ctx.stream);
}


}}  // namespace cv::triton
