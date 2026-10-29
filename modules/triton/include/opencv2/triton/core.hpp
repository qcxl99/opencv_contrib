// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Platform-agnostic cv::triton core API.
//
// This header and the function it declares know nothing about cv::cuda::GpuMat
// or cv::cuda::Stream, and including it does not require any CUDA (or HIP)
// toolkit header to exist. A caller hands over device pointers, sizes and
// strides directly; this is the layer every backend (CUDA, BW1000/HIP, or any
// future one) and every adapter (the NVIDIA GpuMat adapter in triton.hpp, or
// a torch-tensor-backed caller on BW1000) ultimately goes through.
//
// cv::triton::resize(InputArray, OutputArray, ..., cuda::Stream&) in
// triton.hpp is the NVIDIA/OpenCV-GpuMat convenience adapter built ON TOP of
// this core, not the other way around: it validates and unwraps a GpuMat pair
// into the ImageView/Context pair below, then calls resizeCore(). Do not add
// a cuda:: or hip:: type to this header.

#ifndef OPENCV_TRITON_CORE_HPP
#define OPENCV_TRITON_CORE_HPP

#include <cstddef>
#include <cstdint>

#include "opencv2/core.hpp"

namespace cv { namespace triton {

//! @addtogroup triton
//! @{

/** @brief Describes one device-resident image buffer for the core API.
 *
 * Carries byte pitch, not element stride, because that is what device
 * pointer arithmetic needs. whole_* records the parent allocation so a ROI
 * view addresses correctly (both backends already key off the real step,
 * the same contract a ROI GpuMat reports) -- set whole_rows/whole_cols equal
 * to rows/cols and offset_x/offset_y to 0 for a non-ROI/top-level buffer.
 *
 * This is the SAME shape as the module-private backend::ImageView; it is
 * duplicated here (rather than including the backend-private header) so this
 * is a stable public surface independent of backend internals. The two are
 * kept in sync by hand -- see detail::toBackendView in core_resize.cpp.
 */
struct CV_EXPORTS_W_SIMPLE ImageView
{
    CV_PROP_RW std::uintptr_t ptr;      //!< device pointer, as an integer
    CV_PROP_RW int rows;
    CV_PROP_RW int cols;
    CV_PROP_RW int type;                //!< OpenCV type, e.g. CV_8UC3
    CV_PROP_RW std::size_t step;        //!< row pitch IN BYTES
    CV_PROP_RW int whole_rows;          //!< parent allocation height
    CV_PROP_RW int whole_cols;          //!< parent allocation width
    CV_PROP_RW int offset_y;            //!< ROI offset within the parent
    CV_PROP_RW int offset_x;

    ImageView()
        : ptr(0), rows(0), cols(0), type(0), step(0),
          whole_rows(0), whole_cols(0), offset_y(0), offset_x(0) {}

    bool isRoi() const
    {
        return offset_x != 0 || offset_y != 0 ||
               whole_rows != rows || whole_cols != cols;
    }
    int channels() const { return CV_MAT_CN(type); }
    int depth() const { return CV_MAT_DEPTH(type); }
};

/** @brief Device/stream handle for the core API, as plain integers.
 *
 * `device` and `stream` must identify the context that owns src/dst. Neither
 * this struct nor resizeCore() creates a device context of its own: that is
 * the caller's (or the adapter's) responsibility, same as the private
 * backend::Context it maps onto.
 */
struct CV_EXPORTS_W_SIMPLE ExecutionContext
{
    CV_PROP_RW int device;              //!< platform device ordinal
    CV_PROP_RW std::uintptr_t stream;   //!< platform stream handle, may be 0

    ExecutionContext() : device(-1), stream(0) {}
};

/** @brief Resize an image using a Triton-generated kernel -- core entry point.
 *
 * This performs every check cv::triton::resize(GpuMat...) performs (depth,
 * channels, interpolation, explicit dsize, in-place rejection, overlap
 * rejection -- ROI src/dst are accepted, not rejected) against the
 * ImageView/ExecutionContext pair directly, with no GpuMat, no cuda::Stream
 * and no CUDA/HIP header dependency. `dst` must already be allocated to
 * `dsize` by the caller: unlike
 * the GpuMat adapter, this layer never allocates (it has no OutputArray to
 * allocate through).
 *
 * @param src   Source image view, CV_8U or CV_32F, 1/3/4 channels, ROI
 *              views accepted.
 * @param dst   Destination image view, already sized to dsize, ROI views
 *              accepted, not aliased with src.
 * @param dsize Explicit output size. Must equal dst's size.
 * @param interpolation INTER_NEAREST, INTER_LINEAR or INTER_CUBIC.
 * @param ctx   Device and stream that own src/dst.
 * @param scaleX Kernel-space horizontal scale (src cols per dst pixel). 0
 *               derives it as src.cols / dsize.width. A caller resolving a
 *               scale-factor (fx/fy) request must pass 1/fx here instead
 *               of letting it be re-derived from the rounded dsize --
 *               those two values can differ (see cv::triton::resize's
 *               GpuMat adapter), and cv::cuda::resize's own fx/fy path
 *               does NOT re-derive, so bit-exactness with it requires the
 *               same choice here.
 * @param scaleY Kernel-space vertical scale (src rows per dst pixel). 0
 *               derives it as src.rows / dsize.height; see scaleX.
 */
CV_EXPORTS_W void resizeCore(const ImageView& src, const ImageView& dst, Size dsize,
                             int interpolation, const ExecutionContext& ctx,
                             double scaleX = 0, double scaleY = 0);

//! @}

}}  // namespace cv::triton

#endif  // OPENCV_TRITON_CORE_HPP
