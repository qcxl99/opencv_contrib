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

/** @brief Release the module's resources that are held on behalf of captured CUDA graphs.
 *
 * Every operation here can be recorded into a CUDA graph: call it while a stream is capturing
 * and the launch is recorded, not executed. A captured graph refers to the compiled module and
 * to its per-operation tables by handle and may be launched much later, so at capture time the
 * runtime pins those resources so a later launch cannot read an unloaded module or a recycled
 * table. CUDA gives no notification when a graph is destroyed, so the pin is released here.
 *
 * **Contract:** call this once the graphs you captured through this module are destroyed or no
 * longer used. Releasing pins while a graph that reads them can still be launched is a
 * use-after-free that no API can detect.
 *
 * **Cost of not calling it:** pins grow with the number of distinct compiled kernels and table
 * sets you captured -- not with the number of replays or captures of the same kernel -- and are
 * a few kilobytes each. A process that captures a fixed set of operations once and replays them
 * therefore has a fixed, small pin set; one that captures many distinct operation shapes
 * (different dtypes, sizes, interpolation modes, colour codes) should release when done.
 *
 * The module does not allocate graph objects, does not own a graph and cannot tell a live graph
 * from a destroyed one; this is the whole of its graph lifecycle responsibility. Thread-safe.
 */
CV_EXPORTS_W void releaseGraphResources();

/** @brief Resize an image using a Triton-generated kernel.
 *
 * Parameter meaning and result follow cv::resize (see the module description),
 * but the accepted range is narrower (see the module description) and
 * unsupported combinations raise cv::Error::StsUnsupportedFormat instead of
 * falling back to another implementation. There is no implicit fallback to
 * CPU or to cv::cuda: a call that returns has submitted the computation or a same-size copy.
 *
 * @param src   Source GpuMat, CV_8U or CV_32F, 1/3/4 channels.
 * @param dst   Destination GpuMat. Created if needed; never aliased with src. For
 *              BORDER_TRANSPARENT, initialize it at the requested size/type when previous
 *              values matter: outside taps use its existing pixel with their original weight
 *              under LINEAR/AREA, and outside rounded points remain unchanged under NEAREST.
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

/** @brief Convert an image between colour spaces using a Triton-generated kernel.
 *
 * Computes what OpenCV's CPU `cv::cvtColor` computes — bit-exactly, for the codes listed
 * below. Anything else raises `cv::Error::StsUnsupportedFormat`; there is no implicit
 * fallback to the CPU or to `cv::cuda`.
 *
 * Supported for CV_8U (see the module description for why the CV_8U integer conversions are
 * the exact set):
 *   - channel permutes: BGR2BGRA, BGRA2BGR, BGR2RGBA, RGBA2BGR, BGR2RGB, BGRA2RGBA
 *   - to gray: BGR2GRAY, RGB2GRAY, BGRA2GRAY, RGBA2GRAY
 *   - YCrCb: BGR2YCrCb, RGB2YCrCb, YCrCb2BGR, YCrCb2RGB
 *   - HSV: BGR2HSV, RGB2HSV, BGR2HSV_FULL, RGB2HSV_FULL
 *   - YUV: BGR2YUV, RGB2YUV, YUV2BGR, YUV2RGB
 *
 * **Refused on purpose:** HSV2BGR and every HLS code. OpenCV computes those in float32, and in
 * the characterised build the SIMD body and the scalar tail round differently, so the result
 * for a pixel depends on which of the two processes it — position in the row, and build. The
 * result is well-defined per build and per pixel; what is missing is a value stable across
 * positions and builds for a kernel to match, so these are refused rather than given a
 * tolerance. Also refused: CV_32F, CV_16U, 2-channel images, size-changing codes, and anything
 * not in the list above.
 *
 * @param src   Source GpuMat, CV_8U, 3 or 4 channels as the code requires. ROI accepted.
 * @param dst   Destination GpuMat. Created if needed. Its channel count must be the one the
 *              code produces; the size always equals src's.
 * @param code  A `cv::COLOR_*` value from the supported list.
 * @param stream Work is submitted to this stream and the call returns asynchronously.
 *
 * @note The reference is this build's own `cv::cvtColor`. Each served code is integer
 * arithmetic, so an output pixel is a function of its own input bytes and constant tables --
 * no reduction order, no SIMD width. That is why resize needs a pinned reference configuration
 * and cvtColor does not. Equality to that reference was measured: exhaustively over all 256^3
 * colour triples for the 3-channel codes, by sampled alpha for the 4-channel ones, and by a
 * directed position-independence probe -- not inferred from the word "integer".
 *
 * This is a statement about the build that was verified, not about "the CPU" in general:
 * `cv::cvtColor` dispatches through `cv_hal_cvtBGRtoBGR` / `cvtBGRtoYUV` / `cvtBGRtoHSV`, so an
 * IPP or custom-HAL build, or a later OpenCV with different coefficients, is a different
 * reference and must be re-verified.
 */
CV_EXPORTS void cvtColor(InputArray src, OutputArray dst, int code,
                         cuda::Stream& stream = cuda::Stream::Null());

/** @overload
 *
 * The form exposed to Python:
 * `cv2.triton.cvtColor(gpu_src, code[, dst[, stream]]) -> dst`.
 * As with `resize`, the `InputArray` overload above is C++-only because the bindings
 * generator only adds a GpuMat variant for functions in `cv::cuda`.
 */
CV_EXPORTS_W void cvtColor(const cuda::GpuMat& src, CV_OUT cuda::GpuMat& dst, int code,
                           cuda::Stream& stream = cuda::Stream::Null());

/** @brief Apply an affine transformation to an image using a Triton-generated kernel.
 *
 * Computes what OpenCV's CPU `cv::warpAffine` computes, to the tolerances below. cv::cuda::warpAffine
 * is NOT the reference: this OpenCV's CPU warpAffine evaluates float coordinates with float
 * weights, while the GPU implementation still runs the older fixed-point remap pipeline, so the
 * two differ in the last bits of any sub-pixel map. It is a timing baseline only.
 *
 * Supported: CV_8U, CV_16U and CV_32F; 1/3/4 channels; INTER_NEAREST, INTER_LINEAR and
 * INTER_CUBIC; INTER_AREA aliases INTER_LINEAR. The five standard border modes plus
 * BORDER_TRANSPARENT for NEAREST/LINEAR/AREA and WARP_INVERSE_MAP both ways are served.
 * CUBIC with TRANSPARENT, C2 and CV_16S are refused. Anything else raises
 * `cv::Error::StsUnsupportedFormat`; there is no fallback to the CPU or to cv::cuda.
 *
 * **Refused on purpose:** INTER_LANCZOS4. Its 8x8 weight table is a separate contract, and
 * claiming it behind an unmeasured tolerance is precisely the failure this module avoids.
 *
 * Accuracy (contract warpaffine/0.4.1; measured against the selected reference on the core
 * suite, a deterministic combination sweep over this surface and a random sweep):
 *   - CV_8U: bit-exact (got == ref), no one-level allowance.
 *   - CV_16U: |got - ref| <= 1 on every element; its one-level allowance is unchanged.
 *   - CV_32F: |got - ref| <= min(5e-5 * max(|ref|, 1.0), 1e-4) PER ELEMENT, never normalised by the image
 *     maximum. Against the selected reference the measured worst is 0 non-identical elements of
 *     549,312 compared, so the bound is headroom; the accuracy test reports the counts, max
 *     |diff|, max normalised error and max ULP rather than a bare pass. CUBIC float32 measured
 *     1.8e-7 relative, five orders of magnitude inside the same bound.
 *
 * "CPU cv::warpAffine" here means its VECTOR path for INTER_NEAREST and INTER_LINEAR:
 * selected CPU vector-path semantics. That build computes a column's source coordinate as
 * v_fma(M0, x, (float)(y*M1 + M2)) in the SIMD body but as x*M0 + y*M1 + M2 in the scalar tail,
 * and the tail's length depends on the destination width, so the shared region of two different
 * widths can disagree. Magnitude depends on the interpolation: under INTER_NEAREST an ulp can
 * select the ADJACENT source pixel (measured up to 239 of 255 for CV_8U; fixture: width 40 ->
 * 193, widths 48/64/128 -> 65 at one pixel); under INTER_LINEAR the measured worst over
 * 80 seeds x 4 widths was 1 level, which is a measurement on this reference build and not a
 * general guarantee. CV_16U behaves the same way (NEAREST up to an adjacent pixel, LINEAR 1
 * level). This module computes the vector-path result, so it matches a natural-width CPU call
 * except at those isolated pixels -- visibly wrong under NEAREST. See doc/warpaffine.markdown.
 *
 * INTER_CUBIC is the exception and it is measured, not assumed: cv::warpAffine routes it
 * through genericWarp() -> bicubicVec/bicubicRef, whose column coordinate is
 * M_x + M(0,0)*x evaluated from the ABSOLUTE x with no fma. There is no body/tail split, and
 * over 80 seeds x 4 widths 0 of 320 (seed, width) pairs differ for CV_8U, CV_16U and CV_32F.
 * The CUBIC reference is therefore the natural-width call -- padding would change nothing and
 * applying the rule would hide that. `Triton_WarpAffine.CpuCubicIsWidthInvariant` pins it.
 *
 * @param src   Source GpuMat, CV_8U, CV_16U or CV_32F, 1/3/4 channels. ROI accepted.
 * @param dst   Destination GpuMat. Created if needed; never aliased with src. For
 *              BORDER_TRANSPARENT, initialize it at the requested size/type when previous
 *              values matter: outside taps use its existing pixel with their original weight
 *              under LINEAR/AREA, and outside rounded points remain unchanged under NEAREST.
 * @param M     2x3 CV_32F or CV_64F matrix, interpreted as cv::warpAffine interprets it: the
 *              forward map, inverted internally, unless WARP_INVERSE_MAP is set in `flags`.
 * @param dsize Destination size; empty means the source size.
 * @param flags INTER_NEAREST, INTER_LINEAR, INTER_CUBIC or INTER_AREA (LINEAR alias), optionally OR-ed with
 *              WARP_INVERSE_MAP.
 * @param borderMode BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_REFLECT_101 or
 *              BORDER_WRAP; BORDER_TRANSPARENT is supported for NEAREST/LINEAR/AREA.
 * @param borderValue Used by BORDER_CONSTANT. A Scalar of up to four components, one per
 *              channel, exactly as cv::warpAffine reads it: Scalar(v) means (v, 0, 0, 0), not v
 *              on every channel, and the value keeps full double precision, so a CV_32F border
 *              of 0.5 stays 0.5.
 * @param stream Work is submitted to this stream and the call returns asynchronously.
 */
CV_EXPORTS void warpAffine(InputArray src, OutputArray dst, InputArray M, Size dsize,
                           int flags = INTER_LINEAR,
                           int borderMode = BORDER_CONSTANT,
                           const Scalar& borderValue = Scalar(),
                           cuda::Stream& stream = cuda::Stream::Null());

/** @overload
 *
 * The form exposed to Python:
 * `cv2.triton.warpAffine(gpu_src, M, dsize[, dst[, flags[, borderMode[, borderValue[, stream]]]]]) -> dst`.
 */
CV_EXPORTS_W void warpAffine(const cuda::GpuMat& src, CV_OUT cuda::GpuMat& dst,
                             InputArray M, Size dsize,
                             int flags = INTER_LINEAR,
                             int borderMode = BORDER_CONSTANT,
                             const Scalar& borderValue = Scalar(),
                             cuda::Stream& stream = cuda::Stream::Null());

//! @}

}}  // namespace cv::triton

#endif  // OPENCV_TRITON_HPP
