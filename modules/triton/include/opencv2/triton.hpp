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
 * process into a verified on-disk cache; the launch path is C++. Only a cache miss starts
 * the Python compile helper.
 *
 * The module builds for Linux with CUDA or HIP. This header exposes CUDA GpuMat APIs
 * (including their Python wrappers); HIP builds refuse those calls. C++ device-buffer
 * entries for both backends are declared in opencv2/triton/device.hpp. See the module README
 * for each operation's support and backend-specific verification scope.
 *
 * Resize semantics: OpenCV CPU cv::resize as computed by a pinned reference configuration (OpenCV
 * 5.0.0 generic resize.cpp path, 128-bit SIMD, no FMA, no IPP, no HAL): CV_8U bit-exact, CV_32F
 * within a bound derived from that configuration's float32 rounding. cv::cuda::resize computes
 * a different result; this is NOT a drop-in replacement for it.
 *
 * Resize subset (anything else raises, nothing falls back):
 *   - CV_8U or CV_32F, 1, 3 or 4 channels, ROI source/destination, row-contiguous pixels
 *   - explicit dsize, or fx/fy (> 0) when dsize is empty -- dsize wins if both are given, and
 *     given fx/fy are used unchanged as the inverse scales, as in cv::resize
 *   - routes: same size (a device copy), INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA
 *     at integer or fractional ratios, including AREA's upscaled-axis route.
 *   - CUDA GpuMat here; CUDA/HIP device-buffer variants in device.hpp.
 *
 * @see @ref tutorial_triton_h20_first_use "H20 build, install and first-use walkthrough".
 * @see @ref tutorial_triton_resize "Triton resize usage guide" for resize details.
 */

namespace cv { namespace triton {

//! @addtogroup triton
//! @{

/** @brief Is a usable device present for the configured Triton backend?
 *
 * CUDA checks OpenCV's current device; HIP checks visible device 0. A true result does not
 * imply that CUDA GpuMat/Python operations work on HIP; use haveDevice(device) for an explicit
 * device query and the device-buffer C++ entries there. Pure query: no allocation,
 * no submission, no compilation. Whether the compile toolchain is present matters only on a
 * cache miss, and is reported then (an exception), not here.
 */
CV_EXPORTS_W bool haveBackend();

/** @brief Name of the active backend, or an empty string if none. */
CV_EXPORTS_W String backendName();

/** @brief Release the module's resources that are held on behalf of captured backend graphs.
 *
 * The CUDA GpuMat and CUDA/HIP device-buffer operations support capture on a warmed stream
 * within the backend's documented limitations. Call while that backend stream is capturing
 * and the launch is recorded, not executed. A captured graph refers to the compiled module and
 * to its per-operation tables by handle and may be launched much later, so at capture time the
 * runtime pins those resources so a later launch cannot read an unloaded module or a recycled
 * table. The module receives no graph-destruction notification, so pins are released here.
 *
 * **Contract:** call this once the graphs you captured through this module are destroyed or no
 * longer used and their in-flight executions have finished. This releases this runtime's
 * graph pins across contexts, so all affected graphs must satisfy that condition.
 * Releasing pins while a graph that reads them can still be launched is a
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
 * @param dst   Destination GpuMat. Created if needed; correctly sized/type ROI storage is
 *              reused. Source and destination must not alias or overlap.
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
CV_EXPORTS void resize(const cuda::GpuMat& src, CV_OUT cuda::GpuMat& dst, Size dsize,
                         double fx = 0, double fy = 0,
                         int interpolation = INTER_LINEAR,
                         cuda::Stream& stream = cuda::Stream::Null());

/** @brief Convert an image between colour spaces using a Triton-generated kernel.
 *
 * The native module serves 84 distinct numeric codes (aliases share a code):
 * channel permutations (six), colour to gray (four), gray to colour (two),
 * YCrCb (four), YUV (four), forward HSV/HSV_FULL (four), BGR565/BGR555
 * packing/unpacking (twenty), Bayer bilinear (six), XYZ (four), Lab (eight),
 * Luv (six), UYVY/YUY2/YVYU decode (six), YUV gray extraction (two), and
 * NV12/NV21 decode (eight). See the module README for code names and families.
 *
 * CV_8U covers every served code. CV_16U covers the 20 numeric codes in
 * colour/gray, YCrCb/YUV and Bayer bilinear. CV_32F covers 20 numeric codes:
 * all channel permutations, gray to colour, colour to gray, and YCrCb/YUV.
 * The two YUV gray extraction codes are CV_8U-only, as in CPU cv::cvtColor. Copies and integer conversions are exact
 * against the characterised build's CPU cv::cvtColor. For finite 32F arithmetic:
 * `|got-ref| <= 8*2^-24*max(1, |ref|, max(abs(input channels at this pixel)))`.
 * This is a per-pixel scale bound, not whole-image scale or bit-exact arithmetic.
 * IPP/custom-HAL and changed CPU builds require their own validation.
 *
 * Packed BGR565/555 uses CV_8UC2 (two bytes per pixel), not CV_16UC1.
 * 4:2:2 conversions require even width; the two YUV gray extractions allow
 * odd widths. NV12/NV21 use a packed CV_8UC1 source of (3H/2) rows and even W,
 * with source rows divisible by three; output is H by W. Other codes preserve
 * size. Reverse HSV, HLS, forward linear Luv, Bayer VNG/edge-aware, YUV encoding,
 * unlisted codes and unsupported depths are refused without fallback.
 *
 * @param src Source GpuMat, with the depth and channels required by the code.
 *            ROI and row pitch are accepted for interleaved pixels.
 * @param dst Destination GpuMat. Reused when size/type match, otherwise created
 *            at the source depth, code-specific channels and output size.
 *            Source/output overlap is not supported.
 * @param code A supported cv::COLOR_* value. No dstCn override is provided.
 * @param stream Work is submitted asynchronously to this stream. Callers order
 *               producers/consumers and retain image buffers until completion.
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
CV_EXPORTS void cvtColor(const cuda::GpuMat& src, CV_OUT cuda::GpuMat& dst, int code,
                         cuda::Stream& stream = cuda::Stream::Null());

/** @internal Nullable Python stream bridge; HIP raises inside the wrapped call. */
CV_EXPORTS_AS(cvtColor) void cvtColorForPython(const cuda::GpuMat& src, CV_OUT cuda::GpuMat& dst,
                                             int code, Ptr<cuda::Stream> stream = Ptr<cuda::Stream>());

/** @brief Apply an affine transformation to an image using a Triton-generated kernel.
 *
 * Computes what OpenCV's CPU `cv::warpAffine` computes, to the tolerances below. cv::cuda::warpAffine
 * is NOT the reference: this OpenCV's CPU warpAffine evaluates float coordinates with float
 * weights, while the GPU implementation still runs the older fixed-point remap pipeline, so the
 * two differ in the last bits of any sub-pixel map. It is a timing baseline only.
 *
 * Supported: CV_8U, CV_16U and CV_32F; 1/3/4 channels; INTER_NEAREST, INTER_LINEAR and
 * INTER_CUBIC, with INTER_AREA as an alias for INTER_LINEAR; the five standard border modes
 * plus BORDER_TRANSPARENT for NEAREST/LINEAR/AREA; and WARP_INVERSE_MAP both ways. Anything else raises
 * `cv::Error::StsUnsupportedFormat`; there is no fallback to the CPU or to cv::cuda.
 *
 * **Refused on purpose:** INTER_LANCZOS4. Its 8x8 weight table is a separate contract, and
 * claiming it behind an unmeasured tolerance is precisely the failure this module avoids.
 * INTER_CUBIC with BORDER_TRANSPARENT is also refused.
 *
 * BORDER_TRANSPARENT uses the existing destination: NEAREST leaves out-of-range samples
 * unchanged; LINEAR/AREA replace each out-of-range tap with that destination pixel and keep
 * the original weights. A wholly outside neighbourhood leaves the destination unchanged.
 * Supply a destination of the requested size and type with initialized contents before calling;
 * allocation of an empty destination does not provide a defined background. INTER_AREA has
 * LINEAR sampling semantics here, rather than resize-style area integration.
 *
 * Accuracy against the selected reference:
 *   - CV_8U: bit-exact (got == ref).
 *   - CV_16U: |got - ref| <= 1 on every element.
 *   - CV_32F: |got - ref| <= min(5e-5 * max(|ref|, 1.0), 1e-4) per element,
 *     never normalised by the image maximum.
 * Tests report counts, maximum absolute/normalised error and ULP separately from passing.
 *
 * The fused/separate multiply-add choice is calibrated from fixed probes of the actual linked
 * CPU implementation, per depth/channel count and cv::useOptimized() state. Hardware FMA
 * capability and the module's compiler flags do not identify that reference path. This bounded
 * first-use reference identification does not compute the user's output on the CPU.
 *
 * "CPU cv::warpAffine" here means its VECTOR path for INTER_NEAREST/INTER_LINEAR/INTER_AREA:
 * selected CPU vector-path semantics. That build computes a column's source coordinate as
 * v_fma(M0, x, (float)(y*M1 + M2)) in the SIMD body but as x*M0 + y*M1 + M2 in the scalar tail,
 * and the tail's length depends on the destination width, so the shared region of two different
 * widths can disagree. Under NEAREST an ulp at a rounding tie may select an adjacent
 * source pixel, so no small-output-error guarantee is offered against a natural-width call.
 * LINEAR can also differ; the numerical bounds above apply only to the selected reference.
 * The padded reference is tied to the linked CPU build, not a cross-ISA guarantee.
 *
 * INTER_CUBIC uses the natural-width CPU call through genericWarp()/bicubicVec/bicubicRef.
 * Its absolute-column coordinate expression differs from the NN/LINEAR invoker above.
 * See doc/warpaffine.markdown and the CPU width-invariance controls in the tests.
 *
 * @param src   Source GpuMat, CV_8U, CV_16U or CV_32F, 1/3/4 channels. ROI accepted.
 * @param dst   Destination GpuMat. Created if needed; never aliased with src. For
 *              BORDER_TRANSPARENT, supply initialized contents of the required size/type.
 * @param M     2x3 CV_32F or CV_64F matrix, interpreted as cv::warpAffine interprets it: the
 *              forward map, inverted internally, unless WARP_INVERSE_MAP is set in `flags`.
 * @param dsize Destination size; empty means the source size.
 * @param flags INTER_NEAREST, INTER_LINEAR, INTER_CUBIC or INTER_AREA (LINEAR alias), optionally OR-ed with
 *              WARP_INVERSE_MAP.
 * @param borderMode BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_REFLECT_101 or
 *              BORDER_WRAP, or BORDER_TRANSPARENT (NEAREST/LINEAR/AREA only).
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
CV_EXPORTS void warpAffine(const cuda::GpuMat& src, CV_OUT cuda::GpuMat& dst,
                             InputArray M, Size dsize,
                             int flags = INTER_LINEAR,
                             int borderMode = BORDER_CONSTANT,
                             const Scalar& borderValue = Scalar(),
                             cuda::Stream& stream = cuda::Stream::Null());

/** @brief Python binding bridge. A nullable stream avoids constructing a CUDA
 * Stream before the generated wrapper enters its exception handler on HIP.
 * The Python name and argument order remain resize(..., stream=None).
 */
CV_EXPORTS_AS(resize) void resizeForPython(const cuda::GpuMat& src, CV_OUT cuda::GpuMat& dst, Size dsize,
                                         double fx = 0, double fy = 0,
                                         int interpolation = INTER_LINEAR,
                                         Ptr<cuda::Stream> stream = Ptr<cuda::Stream>());

/** @brief Python binding bridge with a nullable, safely constructed stream. */
CV_EXPORTS_AS(warpAffine) void warpAffineForPython(const cuda::GpuMat& src, CV_OUT cuda::GpuMat& dst,
                                                 InputArray M, Size dsize,
                                                 int flags = INTER_LINEAR,
                                                 int borderMode = BORDER_CONSTANT,
                                                 const Scalar& borderValue = Scalar(),
                                                 Ptr<cuda::Stream> stream = Ptr<cuda::Stream>());

//! @}

}}  // namespace cv::triton

#endif  // OPENCV_TRITON_HPP
