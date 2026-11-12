// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// The device-pointer entries (opencv2/triton/device.hpp). Backend-neutral: compiled into every
// build. The checks are the GpuMat entries' own (cuda_resize_adapter.cpp), stated again here
// because those entries validate GpuMat arguments this layer does not have; the runtime then
// applies the operation's full normalisation, exactly as for a GpuMat call.

#include "precomp.hpp"
#include "opencv2/triton/device.hpp"
#include "runtime/cvtcolor.hpp"
#include "runtime/warpaffine.hpp"
#include "runtime/runtime.hpp"

namespace cv { namespace triton {

namespace {

kf_resize_rt::DeviceView view(const DeviceImage& m, int device)
{
    if (m.ptr == 0 || m.rows <= 0 || m.cols <= 0)
        CV_Error(Error::StsBadArg, "cv::triton: a device image needs a non-null pointer and a positive size");
    ImageView v;
    v.ptr = m.ptr;
    v.rows = m.rows;
    v.cols = m.cols;
    v.type = m.type;
    v.step = m.step;
    v.whole_rows = m.rows;
    v.whole_cols = m.cols;
    // The owning allocation is unknown here (0, 0): the runtime's overlap check then uses the
    // two views' own byte ranges, which is what a caller-owned buffer pair can promise.
    return detail::toDeviceView(v, device, 0, 0);
}

void requireDevice(int device, const char* op)
{
    if (!runtime::deviceAvailable(device))
        CV_Error(Error::StsNotImplemented,
                 cv::format("cv::triton::%s: no usable %s device %d; this module never falls back "
                            "to CPU or cv::cuda", op, runtime::name(), device));
}

}  // namespace

bool haveDevice(int device)
{
    return runtime::deviceAvailable(device);
}

void resizeDevice(const DeviceImage& src, const DeviceImage& dst, Size dsize,
                  int interpolation, int device, std::uintptr_t stream)
{
    if (CV_MAT_DEPTH(src.type) != CV_8U && CV_MAT_DEPTH(src.type) != CV_32F)
        CV_Error(Error::StsUnsupportedFormat,
                 cv::format("cv::triton::resize: depth %d is not supported (CV_8U and CV_32F only)",
                            CV_MAT_DEPTH(src.type)));
    if (dsize.width <= 0 || dsize.height <= 0)
        CV_Error(Error::StsBadArg, "cv::triton::resize: dsize must be explicit and non-empty");
    if (dst.rows != dsize.height || dst.cols != dsize.width || dst.type != src.type)
        CV_Error(Error::StsBadArg, "cv::triton::resize: dst must already be dsize with src's type");
    if (src.ptr == dst.ptr)
        CV_Error(Error::StsBadArg, "cv::triton::resize: in-place resize is not supported");
    requireDevice(device, "resize");
    detail::resizeDevice(view(src, device), view(dst, device), dsize, 0.0, 0.0, interpolation,
                         device, stream);
}

void cvtColorDevice(const DeviceImage& src, const DeviceImage& dst, int code,
                    int device, std::uintptr_t stream)
{
    const kf_cvtcolor::CodeSpec* spec = kf_cvtcolor::find(code);
    if (!spec)
        CV_Error(Error::StsUnsupportedFormat,
                 cv::format("cv::triton::cvtColor: code %d is not served by this module", code));
    if (dst.rows != src.rows || dst.cols != src.cols || CV_MAT_CN(dst.type) != spec->dcn)
        CV_Error(Error::StsBadArg,
                 cv::format("cv::triton::cvtColor: dst must be src's size with %d channels", spec->dcn));
    if (src.ptr == dst.ptr)
        CV_Error(Error::StsBadArg, "cv::triton::cvtColor: in-place conversion is not supported");
    requireDevice(device, "cvtColor");
    kf_cvtcolor::Request req;
    req.src = view(src, device);
    req.dst = view(dst, device);
    req.code = code;
    detail::cvtColorDevice(req, device, stream);
}

void warpAffineDevice(const DeviceImage& src, const DeviceImage& dst, const double M[6],
                      int flags, int borderMode, const Scalar& borderValue,
                      int device, std::uintptr_t stream)
{
    if (flags & ~(INTER_MAX | WARP_INVERSE_MAP))
        CV_Error(Error::StsUnsupportedFormat,"cv::triton::warpAffine: unsupported flags");
    if (!M) CV_Error(Error::StsBadArg,"cv::triton::warpAffine: matrix pointer must not be null");
    const int interpolation = flags & INTER_MAX;
    if (interpolation != INTER_NEAREST && interpolation != INTER_LINEAR &&
        interpolation != INTER_CUBIC && interpolation != INTER_AREA)
        CV_Error(Error::StsUnsupportedFormat,
                 cv::format("cv::triton::warpAffine: interpolation %d is not served by this module "
                            "(INTER_NEAREST, INTER_LINEAR, INTER_CUBIC and INTER_AREA); "
                            "INTER_LANCZOS4 is refused, not approximated", interpolation));
    if (interpolation == INTER_CUBIC && borderMode == BORDER_TRANSPARENT)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::warpAffine: BORDER_TRANSPARENT with INTER_CUBIC is not served");
    const int depth = CV_MAT_DEPTH(src.type);
    if (depth != CV_8U && depth != CV_16U && depth != CV_32F)
        CV_Error(Error::StsUnsupportedFormat,
                 cv::format("cv::triton::warpAffine: depth %d is not served by this module "
                            "(CV_8U, CV_16U and CV_32F)", depth));
    if (dst.type != src.type)
        CV_Error(Error::StsBadArg, "cv::triton::warpAffine: dst must have src's type");
    if (src.ptr == dst.ptr)
        CV_Error(Error::StsBadArg, "cv::triton::warpAffine: in-place operation is not supported");
    requireDevice(device, "warpAffine");
    kf_warpaffine::Request req;
    req.src = view(src, device);
    req.dst = view(dst, device);
    for (int i = 0; i < 6; ++i) req.M[i] = M[i];
    req.interpolation = interpolation;
    req.border_mode = borderMode;
    for (int i = 0; i < 4; ++i) req.border[i] = borderValue[i];
    req.inverse_map = (flags & WARP_INVERSE_MAP) != 0;
    detail::warpAffineDevice(req, device, stream);
}

}}  // namespace cv::triton
