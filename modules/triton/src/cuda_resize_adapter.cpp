// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// NVIDIA/OpenCV-GpuMat adapter over the platform-agnostic core (core_resize.cpp
// / src/core.hpp, module-internal). This file -- not the core -- is what requires
// cuda::GpuMat, cuda::Stream and cuda_runtime.h: it unwraps a GpuMat pair into
// an ImageView/ExecutionContext pair, delegates every validation and the
// actual launch to resizeCore(), and re-wraps the result for the caller
// (allocating dst via OutputArray::create(), which the core layer cannot do
// since it has no OutputArray).
//
// This is the only file in the module that is compiled against the CUDA
// runtime headers (see CMakeLists.txt). The first submission targets Linux +
// CUDA only; a build of this module without it is not offered.

#include "precomp.hpp"
#include "runtime/cvtcolor.hpp"
#include "runtime/warpaffine.hpp"
#include "runtime/runtime.hpp"

namespace cv { namespace triton {

bool haveBackend()
{
    // Pure query: no device memory is allocated and no work is submitted. Whether the
    // compile toolchain is present matters only on a cache miss and is reported then.
    if (cuda::getCudaEnabledDeviceCount() < 1)
        return false;
    return runtime::deviceAvailable(cuda::getDevice());
}

String backendName()
{
    return haveBackend() ? String(runtime::name()) : String();
}

void releaseGraphResources()
{
    // The whole graph lifecycle responsibility of this module: the runtime pins the compiled
    // module and its tables when a launch is recorded into a capturing stream, and cannot know
    // when the caller's graph is destroyed. See the declaration for the contract and the cost.
    detail::releaseCapturePins();
}

namespace {

ImageView toImageView(const cuda::GpuMat& m)
{
    Size whole;
    Point ofs;
    m.locateROI(whole, ofs);

    ImageView v;
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

}  // namespace

void resize(InputArray _src, OutputArray _dst, Size dsize,
            double fx, double fy, int interpolation, cuda::Stream& stream)
{
    CV_Assert(_src.kind() == _InputArray::CUDA_GPU_MAT);
    cuda::GpuMat src = _src.getGpuMat();

    // dsize/fx-fy precedence, matching cv::cuda::resize (cudawarping/src/
    // resize.cpp:74-84) exactly: one of the two forms must be given, and
    // when dsize IS given fx/fy are ignored rather than cross-checked
    // against it (same rule this function already had before fx/fy were
    // accepted, so passing both is not an error, it is just dsize winning).
    //
    // cvRound(double) is saturate_cast<int>(double) is round-half-to-even
    // via SSE2 CVTSD2SI on x86_64 (core/fast_math.hpp, core/saturate.hpp) --
    // the SAME function cv::cuda::resize's own derivation calls
    // (saturate_cast<int>(src.cols * fx)), so this is bit-for-bit the
    // upstream rounding rule, not an approximation of it.
    //
    // Subtlety verified against cudawarping/src/resize.cpp:76-84, NOT
    // assumed: when fx/fy ARE given, upstream does NOT re-derive them from
    // the rounded dsize afterwards -- it keeps the caller's original
    // (possibly inexact) fx/fy and passes 1/fx, 1/fy straight to the
    // kernel. Those two numbers can differ: e.g. fx=0.33 on cols=96 rounds
    // to dsize.width=32, but 1/0.33 = 3.0303... != 96/32 = 3.0 exactly, and
    // the kernel's bilinear weights are sensitive to which one it gets.
    // Re-deriving fx/fy from dsize here (src.cols/dsize.width) would
    // silently disagree with cv::cuda::resize on every non-exact scale
    // factor, which is precisely the case this path exists to exercise --
    // so the ORIGINAL fx/fy (when given) must reach the kernel unchanged,
    // and only a dsize-given call computes kernel-space scale from dsize.
    // resize serves CV_8U and CV_32F only. The shared device-view conversion also accepts
    // CV_16U (warpAffine needs it), so resize states its own depth rule here, at its own entry,
    // before anything is allocated or submitted -- refusing later would let a 16-bit request
    // reach the compile helper, which fails with a different, undocumented error code.
    if (src.depth() != CV_8U && src.depth() != CV_32F)
        CV_Error(Error::StsUnsupportedFormat,
                 cv::format("cv::triton::resize: depth %d is not supported (CV_8U and CV_32F only)",
                            src.depth()));
    const bool fxfyGiven = dsize.empty();
    if (dsize.empty() && !(fx > 0 && fy > 0))
        CV_Error(Error::StsBadArg,
                 "cv::triton::resize: either dsize or both fx and fy (> 0) must be given");
    if (dsize.empty())
        dsize = Size(cvRound(src.cols * fx), cvRound(src.rows * fy));
    if (dsize.width <= 0 || dsize.height <= 0)
        CV_Error(Error::StsBadArg,
                 cv::format("cv::triton::resize: invalid resulting dsize %dx%d",
                            dsize.width, dsize.height));
    // ROI src/dst are supported -- see the note in core_resize.cpp's
    // resizeCore() for why no addressing gate is needed here either.

    // Detect in-place BEFORE _dst.create(): create() may reallocate dst when
    // the requested size differs from its current size, after which
    // dst.cudaPtr() no longer equals src.cudaPtr() and the aliasing is
    // invisible to resizeCore()'s post-create pointer check below. Compare
    // the caller's arrays as given, not the post-create buffers.
    //
    // This check was dropped when this file was split out of the former
    // resize.cpp (see opencv_contrib@b8a2073, which added it originally),
    // and its loss was silent: resizeCore()'s own overlap check is correct
    // on its own terms, it just runs too late to see this specific case.
    // Caught by TritonResize.RejectsUnsupportedInputs's in-place sub-case
    // (`cv::triton::resize(u8c1, u8c1, Size(32, 32))` on a 64x64 u8c1 --
    // dsize differs from src's size, so create() reallocates dst to a new
    // address before resizeCore() ever sees it, and the post-create check
    // then compares two already-different pointers and finds nothing wrong).
    if (_dst.kind() == _InputArray::CUDA_GPU_MAT)
    {
        const cuda::GpuMat preDst = _dst.getGpuMat();
        if (!preDst.empty() && preDst.cudaPtr() == src.cudaPtr())
            CV_Error(Error::StsBadArg,
                     "cv::triton::resize: in-place resize is not supported");
    }

    // --- prepare the output, THEN read its address -------------------------
    // create() may reallocate, so the pointer, step and ownership must be
    // re-read afterwards rather than captured before.
    _dst.create(dsize, src.type());
    cuda::GpuMat dst = _dst.getGpuMat();
    CV_Assert(dst.rows == dsize.height && dst.cols == dsize.width);
    CV_Assert(dst.type() == src.type());

    const int device = cuda::getDevice();
    const std::uintptr_t cuStream = reinterpret_cast<std::uintptr_t>(cuda::StreamAccessor::getStream(stream));
    auto view = [&](const cuda::GpuMat& m) {
        return detail::toDeviceView(toImageView(m), device, reinterpret_cast<std::uintptr_t>(m.datastart),
                                    static_cast<std::size_t>(m.dataend - m.datastart));
    };
    // fx/fy reach the computation unchanged when they were given (cv::resize uses them as
    // the inverse scales); with dsize they are ignored, as in cv::resize.
    detail::resizeDevice(view(src), view(dst), fxfyGiven ? Size() : dsize, fxfyGiven ? fx : 0.0,
                         fxfyGiven ? fy : 0.0, interpolation, device, cuStream);
    // Asynchronous: everything needed is on `stream`. No implicit sync here.
}

void resize(const cuda::GpuMat& src, cuda::GpuMat& dst, Size dsize,
            double fx, double fy, int interpolation, cuda::Stream& stream)
{
    resize(InputArray(src), OutputArray(dst), dsize, fx, fy, interpolation, stream);
}

void cvtColor(InputArray _src, OutputArray _dst, int code, cuda::Stream& stream)
{
    CV_Assert(_src.kind() == _InputArray::CUDA_GPU_MAT);
    cuda::GpuMat src = _src.getGpuMat();

    // The destination channel count is a property of the CODE, not of src: BGR2BGRA writes 4
    // channels from a 3-channel source, BGR2GRAY writes 1. Deriving it here (rather than
    // letting the caller's dst decide) is what makes `dst` optional from Python while still
    // producing a correctly shaped buffer.
    const kf_cvtcolor::CodeSpec* spec = kf_cvtcolor::find(code);
    if (!spec)
        CV_Error(Error::StsUnsupportedFormat,
                 cv::format("cv::triton::cvtColor: code %d is not served by this module; see "
                            "opencv2/triton.hpp for the supported set (the float32 conversions "
                            "are excluded on purpose)", code));

    // In-place is refused for the same reason as in resize(), and detected BEFORE create():
    // create() may reallocate, after which the aliasing is invisible to the overlap check.
    if (_dst.kind() == _InputArray::CUDA_GPU_MAT)
    {
        const cuda::GpuMat preDst = _dst.getGpuMat();
        if (!preDst.empty() && preDst.cudaPtr() == src.cudaPtr())
            CV_Error(Error::StsBadArg, "cv::triton::cvtColor: in-place conversion is not supported");
    }

    // Same type for src and dst here (CV_8U both ways for every supported code); the channel
    // count comes from the code, the size from src.
    _dst.create(src.size(), CV_MAKETYPE(src.depth(), spec->dcn));
    cuda::GpuMat dst = _dst.getGpuMat();
    CV_Assert(dst.rows == src.rows && dst.cols == src.cols);
    CV_Assert(dst.channels() == spec->dcn);

    const int device = cuda::getDevice();
    const std::uintptr_t cuStream = reinterpret_cast<std::uintptr_t>(cuda::StreamAccessor::getStream(stream));
    auto view = [&](const cuda::GpuMat& m) {
        return detail::toDeviceView(toImageView(m), device, reinterpret_cast<std::uintptr_t>(m.datastart),
                                    static_cast<std::size_t>(m.dataend - m.datastart));
    };
    kf_cvtcolor::Request req;
    req.src = view(src);
    req.dst = view(dst);
    req.code = code;
    detail::cvtColorDevice(req, device, cuStream);
    // Asynchronous, as resize is: everything needed is on `stream`.
}

void warpAffine(InputArray _src, OutputArray _dst, InputArray _M, Size dsize, int flags,
                int borderMode, const Scalar& borderValue, cuda::Stream& stream)
{
    CV_Assert(_src.kind() == _InputArray::CUDA_GPU_MAT);
    cuda::GpuMat src = _src.getGpuMat();

    if (flags & ~(INTER_MAX | WARP_INVERSE_MAP))
        CV_Error(Error::StsUnsupportedFormat,"cv::triton::warpAffine: unsupported flags");
    // INTER_AREA follows the CPU's rewrite to INTER_LINEAR in normalize().
    const int interpolation = flags & INTER_MAX;
    if (interpolation != INTER_NEAREST && interpolation != INTER_LINEAR &&
        interpolation != INTER_CUBIC && interpolation != INTER_AREA)
        CV_Error(Error::StsUnsupportedFormat,
                 cv::format("cv::triton::warpAffine: interpolation %d is not served by this "
                            "module (warpaffine/0.4.1 covers INTER_NEAREST, INTER_LINEAR, "
                            "INTER_CUBIC and INTER_AREA); INTER_LANCZOS4 is refused, not "
                            "approximated",
                            interpolation));
    if (interpolation == INTER_CUBIC && borderMode == BORDER_TRANSPARENT)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::warpAffine: BORDER_TRANSPARENT with INTER_CUBIC is not served");
    {
        const int depth = src.depth();
        if (depth != CV_8U && depth != CV_16U && depth != CV_32F)
            CV_Error(Error::StsUnsupportedFormat,
                     cv::format("cv::triton::warpAffine: depth %d is not served by this "
                                "module (warpaffine/0.4.1 covers CV_8U, CV_16U and CV_32F)",
                                depth));
    }

    // The matrix is read here, in double, exactly as cv::warpAffine reads it: CV_32F or
    // CV_64F, 2x3. The runtime converts and inverts; nothing above this line changes it.
    Mat M = _M.getMat();
    CV_Assert(M.rows == 2 && M.cols == 3 && (M.type() == CV_32F || M.type() == CV_64F));

    if (_dst.kind() == _InputArray::CUDA_GPU_MAT)
    {
        const cuda::GpuMat preDst = _dst.getGpuMat();
        if (!preDst.empty() && preDst.cudaPtr() == src.cudaPtr())
            CV_Error(Error::StsBadArg, "cv::triton::warpAffine: in-place operation is not supported");
    }

    // The destination type is always src's: warpAffine never changes depth or channel count,
    // unlike cvtColor. dsize empty means "same size as src", as in cv::warpAffine.
    _dst.create(dsize.empty() ? src.size() : dsize, src.type());
    cuda::GpuMat dst = _dst.getGpuMat();

    const int device = cuda::getDevice();
    const std::uintptr_t cuStream = reinterpret_cast<std::uintptr_t>(cuda::StreamAccessor::getStream(stream));
    auto view = [&](const cuda::GpuMat& m) {
        return detail::toDeviceView(toImageView(m), device, reinterpret_cast<std::uintptr_t>(m.datastart),
                                    static_cast<std::size_t>(m.dataend - m.datastart));
    };
    kf_warpaffine::Request req;
    req.src = view(src);
    req.dst = view(dst);
    Mat m64;
    M.convertTo(m64, CV_64F);
    for (int i = 0; i < 6; ++i) req.M[i] = m64.at<double>(i / 3, i % 3);
    req.interpolation = interpolation;
    req.border_mode = borderMode;
    // The Scalar is carried per channel: cv2.warpAffine(..., borderValue=(10,20,30)) fills
    // (10,20,30), and a single scalar fills (v,0,0). Reducing it to borderValue[0] is wrong
    // for every channel but 0. It stays double here; nothing narrows it before the kernel.
    for (int i = 0; i < 4; ++i) req.border[i] = borderValue[i];
    req.inverse_map = (flags & WARP_INVERSE_MAP) != 0;
    detail::warpAffineDevice(req, device, cuStream);
    // Asynchronous, as resize is.
}

void cvtColor(const cuda::GpuMat& src, cuda::GpuMat& dst, int code, cuda::Stream& stream)
{
    cvtColor(InputArray(src), OutputArray(dst), code, stream);
}

void warpAffine(const cuda::GpuMat& src, cuda::GpuMat& dst, InputArray M, Size dsize, int flags,
                int borderMode, const Scalar& borderValue, cuda::Stream& stream)
{
    warpAffine(InputArray(src), OutputArray(dst), M, dsize, flags, borderMode, borderValue, stream);
}

}}  // namespace cv::triton
