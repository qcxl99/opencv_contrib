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

}}  // namespace cv::triton
