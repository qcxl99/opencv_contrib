// NVIDIA/OpenCV-GpuMat adapter over the platform-agnostic core (core_resize.cpp
// / opencv2/triton/core.hpp). This file -- not the core -- is what requires
// cuda::GpuMat, cuda::Stream and cuda_runtime.h: it unwraps a GpuMat pair into
// an ImageView/ExecutionContext pair, delegates every validation and the
// actual launch to resizeCore(), and re-wraps the result for the caller
// (allocating dst via OutputArray::create(), which the core layer cannot do
// since it has no OutputArray).
//
// Only compiled when TRITON_BACKEND_CUDA is selected (see CMakeLists.txt). A
// HIP/other-backend build compiles gpumat_adapter_stub.cpp INSTEAD of this
// file, never both -- that stub, not an #ifdef here, is what makes
// haveBackend()/backendName()/resize(GpuMat...) degrade safely when there is
// no CUDA adapter in the binary.

#include "precomp.hpp"

namespace cv { namespace triton {

bool haveBackend()
{
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
}

String backendName()
{
    return haveBackend() ? String(backend::name()) : String();
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

    // Explicit dsize only. fx/fy derive dsize through a different rounding
    // rule (saturate_cast<int>(src*scale)); accepting them silently would
    // change results, so they are rejected while unsupported. Checked here
    // (adapter-side) because resizeCore() takes dsize as already-resolved and
    // has no fx/fy concept of its own.
    if (dsize.empty())
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::resize: explicit non-empty dsize is required; the fx/fy "
                 "scale-factor path is not supported in this round");
    if (fx != 0 || fy != 0)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::resize: fx/fy must be 0 while only explicit dsize is supported");

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

    ExecutionContext ctx;
    ctx.device = cuda::getDevice();
    ctx.stream = reinterpret_cast<std::uintptr_t>(cuda::StreamAccessor::getStream(stream));

    resizeCore(toImageView(src), toImageView(dst), dsize, interpolation, ctx);
    // Asynchronous: everything needed is on `stream`. No implicit sync here.
}

}}  // namespace cv::triton
