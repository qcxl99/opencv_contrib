// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Triton-backed image operations for OpenCV. PROPOSED module, not yet upstream API.
// See the module README for a quick start and doc/resize.markdown for the full
// usage guide; this header is the API reference.
//
// Deliberately SEPARATE from cv::cuda. Calling cv::cuda::resize never reaches
// this module: the caller opts in explicitly by calling cv::triton::resize.
// That keeps the native implementation available as an independent performance
// reference and means this module cannot regress existing behaviour.

#ifndef OPENCV_TRITON_HPP
#define OPENCV_TRITON_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgproc.hpp"   // INTER_LINEAR / INTER_NEAREST

/** @defgroup triton Triton-backed operations
 *
 * Kernels are written in Triton and compiled on first use by a separate, Torch-free helper
 * process into a verified on-disk cache; the call path is C++ and never enters Python.
 *
 * Semantics: OpenCV CPU cv::resize as computed by a pinned reference configuration (OpenCV
 * 5.0.0 generic resize.cpp path, 128-bit SIMD, no FMA, no IPP, no HAL): CV_8U bit-exact, CV_32F
 * within a bound derived from that configuration's float32 rounding. cv::cuda::resize computes
 * a different result; this is NOT a drop-in replacement for it.
 *
 * Supported subset of this milestone (anything else raises, nothing falls back):
 *   - CV_8U or CV_32F, 1, 3 or 4 channels, ROI source/destination, row-contiguous pixels
 *   - explicit dsize, or fx/fy (> 0) when dsize is empty -- dsize wins if both are given, and
 *     given fx/fy are used unchanged as the inverse scales, as in cv::resize
 *   - routes: same size (a device copy), INTER_NEAREST, INTER_LINEAR, INTER_AREA at integer
 *     ratios or with an upscaled axis. INTER_CUBIC and fractional INTER_AREA raise
 *     StsUnsupportedFormat ("route_not_enabled") until the next milestone.
 *   - Linux, CUDA.
 *
 * @see @ref tutorial_triton_resize "Triton resize usage guide" for build instructions and examples.
 */

namespace cv { namespace triton {

//! @addtogroup triton
//! @{

/** @brief Is a usable Triton backend present for the current device?
 *
 * False means no usable CUDA device for the current device index. Pure query: no allocation,
 * no submission, no compilation. Whether the compile toolchain is present matters only on a
 * cache miss, and is reported then (an exception), not here.
 */
CV_EXPORTS_W bool haveBackend();

/** @brief Name of the active backend, or an empty string if none. */
CV_EXPORTS_W String backendName();

/** @brief Resize an image using a Triton-generated kernel.
 *
 * Parameter meaning and result follow cv::resize (see the module description),
 * but the accepted range is narrower (see the module description) and
 * unsupported combinations raise cv::Error::StsUnsupportedFormat instead of
 * falling back to another implementation. There is no implicit fallback to
 * CPU or to cv::cuda: a call that returns has submitted the computation or a same-size copy.
 *
 * @param src   Source GpuMat, CV_8U or CV_32F, 1/3/4 channels.
 * @param dst   Destination GpuMat. Created if needed; never aliased with src.
 * @param dsize Explicit output size, or an empty Size() to derive it from
 *              fx/fy instead.
 * @param fx    Horizontal scale factor, used only when dsize is empty.
 *              dst width is cvRound(src.cols * fx).
 * @param fy    Vertical scale factor, used only when dsize is empty.
 *              dst height is cvRound(src.rows * fy).
 * @param interpolation INTER_NEAREST, INTER_LINEAR, INTER_CUBIC or INTER_AREA.
 * @param stream Work is submitted to this stream and the call returns
 *               asynchronously. Inputs, outputs and any temporaries must stay
 *               alive until the stream completes.
 */
CV_EXPORTS void resize(InputArray src, OutputArray dst, Size dsize,
                       double fx = 0, double fy = 0,
                       int interpolation = INTER_LINEAR,
                       cuda::Stream& stream = cuda::Stream::Null());

/** @overload
 *
 * Same operation for `cuda::GpuMat` arguments, and the form exposed to Python
 * (`cv2.triton.resize(gpu_src, dsize[, dst[, fx[, fy[, interpolation[, stream]]]]]) -> dst`).
 * The InputArray overload above is C++-only: the Python bindings generator only
 * adds a GpuMat variant for functions in `cv::cuda`, so exporting that one gave
 * Python a Mat/UMat signature that this CUDA-only module rejects at run time.
 */
CV_EXPORTS_W void resize(const cuda::GpuMat& src, CV_OUT cuda::GpuMat& dst, Size dsize,
                         double fx = 0, double fy = 0,
                         int interpolation = INTER_LINEAR,
                         cuda::Stream& stream = cuda::Stream::Null());

//! @}

}}  // namespace cv::triton

#endif  // OPENCV_TRITON_HPP
