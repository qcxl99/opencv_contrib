// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_TRITON_HIP_HPP
#define OPENCV_TRITON_HIP_HPP

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <memory>

namespace cv { namespace triton {

//! @addtogroup triton
//! @{

/** @brief An owning HIP device image, available in Python as cv2.triton.HipImage.
 *
 * Stores a pitched 2-D image on the selected device. upload() and download() are
 * blocking NumPy/Mat convenience transfers. Operations accepting HipImage enqueue
 * device work and return HipImage; no host image transfer occurs between operators.
 * Shared storage, including ROIs, tracks outstanding reads and writes across streams.
 * Releasing its last owner waits for outstanding work before freeing storage.
 * See @ref tutorial_triton_hip_python.
 */
class CV_EXPORTS_W HipImage
{
public:
    /** @brief Construct an empty image on a HIP device.
     * @param device Visible HIP device index; the caller's current context is preserved.
     */
    CV_WRAP explicit HipImage(int device = 0);
    ~HipImage();
    HipImage(const HipImage&) = delete;
    HipImage& operator=(const HipImage&) = delete;

    /** @brief Copy a host image to this device, allocating or reusing storage.
     * @param src Non-empty 2-D CV_8U/CV_16U/CV_32F image with 1 to 4 channels.
     * Strided host images are accepted. Matching ROI storage is reused.
     */
    CV_WRAP void upload(InputArray src);
    /** @brief Wait for pending work and copy to a host image (NumPy array in Python).
     * @param dst Output host image of this image's size and type.
     */
    CV_WRAP void download(OutputArray dst) const;
    /** @brief Return a view sharing storage and synchronization with this image.
     * @param rect Non-empty rectangle contained in this image.
     * The view remains valid after the parent object is released.
     */
    CV_WRAP Ptr<HipImage> roi(Rect rect) const;
    CV_WRAP bool empty() const;
    CV_WRAP Size size() const;
    CV_WRAP int type() const;
    CV_WRAP int channels() const;
    CV_WRAP int depth() const;
    CV_WRAP int device() const;
    //! Row pitch in bytes; zero for an empty image.
    CV_WRAP size_t step() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl;
    friend struct HipAccess;
};

/** @brief An owning non-blocking HIP stream, available as cv2.triton.HipStream.
 * All images used with the stream must belong to its device. Destruction waits
 * for queued work. HIP graph capture is not supported by the owning image API.
 */
class CV_EXPORTS_W HipStream
{
public:
    /** @param device Visible HIP device index; the caller's current context is preserved. */
    CV_WRAP explicit HipStream(int device = 0);
    ~HipStream();
    HipStream(const HipStream&) = delete;
    HipStream& operator=(const HipStream&) = delete;
    CV_WRAP int device() const;
    //! Block the host until queued work completes.
    CV_WRAP void waitForCompletion();
    //! Return whether queued work is complete, without blocking the host.
    CV_WRAP bool queryIfComplete() const;
    /** @brief Make subsequent work wait for work already queued in producer.
     * @param producer Stream on the same device. Does not block the host.
     * Image dependencies are also inserted automatically by the operators.
     */
    CV_WRAP void waitFor(const Ptr<HipStream>& producer);

private:
    struct Impl;
    std::shared_ptr<Impl> impl;
    friend struct HipAccess;
};

/** @brief Resize a HIP image and return device-resident output.
 * @param src Source HipImage. Types and interpolation match cv::triton::resize.
 * @param dsize Output size, or (0, 0) to use fx/fy.
 * @param dst Optional output HipImage on src's device; allocated or reused. Must not overlap src.
 * @param fx Horizontal scale; used only with empty dsize.
 * @param fy Vertical scale; used only with empty dsize.
 * @param interpolation INTER_NEAREST, INTER_LINEAR, INTER_CUBIC or INTER_AREA.
 * @param stream Optional HipStream on src's device; null selects its default stream.
 */
CV_EXPORTS Ptr<HipImage> resizeHip(const Ptr<HipImage>& src, Size dsize,
    Ptr<HipImage> dst = Ptr<HipImage>(), double fx = 0, double fy = 0,
    int interpolation = INTER_LINEAR, Ptr<HipStream> stream = Ptr<HipStream>());

/** @brief Convert a HIP image and return device-resident output.
 * @param src Source HipImage with the code's required depth/channels.
 * @param code Supported cv::COLOR_* value; see @ref tutorial_triton_cvtcolor.
 * @param dst Optional output on src's device; allocated or reused. Must not overlap src.
 * @param stream Optional HipStream on src's device; null selects its default stream.
 * Output size and type follow cv::triton::cvtColor.
 */
CV_EXPORTS Ptr<HipImage> cvtColorHip(const Ptr<HipImage>& src, int code,
    Ptr<HipImage> dst = Ptr<HipImage>(), Ptr<HipStream> stream = Ptr<HipStream>());

/** @brief Transform a HIP image and return device-resident output.
 * @param src Source HipImage. Types and sampling follow cv::triton::warpAffine.
 * @param M 2x3 CV_32F/CV_64F forward matrix; WARP_INVERSE_MAP selects inverse mapping.
 * @param dsize Output size; empty selects src.size().
 * @param dst Optional output on src's device. Must not overlap src. BORDER_TRANSPARENT
 * requires an initialized output of matching size/type, from upload() or an earlier operation.
 * @param flags INTER_NEAREST/LINEAR/CUBIC/AREA, optionally combined with WARP_INVERSE_MAP.
 * @param borderMode BORDER_CONSTANT/REPLICATE/REFLECT/REFLECT_101/WRAP/TRANSPARENT.
 * CUBIC with TRANSPARENT is unsupported.
 * @param borderValue Constant-border value per channel.
 * @param stream Optional HipStream on src's device; null selects its default stream.
 */
CV_EXPORTS Ptr<HipImage> warpAffineHip(const Ptr<HipImage>& src, InputArray M,
    Size dsize, Ptr<HipImage> dst = Ptr<HipImage>(), int flags = INTER_LINEAR,
    int borderMode = BORDER_CONSTANT, const Scalar& borderValue = Scalar(),
    Ptr<HipStream> stream = Ptr<HipStream>());

/** @brief Resize a HIP device image; returns the supplied or allocated HipImage.
 * See resizeHip for parameters and supported inputs.
 */
CV_EXPORTS_AS(resize) Ptr<HipImage> resizeForPython(const Ptr<HipImage>& src, Size dsize,
    Ptr<HipImage> dst = Ptr<HipImage>(), double fx = 0, double fy = 0,
    int interpolation = INTER_LINEAR, Ptr<HipStream> stream = Ptr<HipStream>());
/** @brief Convert a HIP device image; returns the supplied or allocated HipImage.
 * See cvtColorHip for supported conversions and output layout.
 */
CV_EXPORTS_AS(cvtColor) Ptr<HipImage> cvtColorForPython(const Ptr<HipImage>& src, int code,
    Ptr<HipImage> dst = Ptr<HipImage>(), Ptr<HipStream> stream = Ptr<HipStream>());
/** @brief Transform a HIP device image; returns the supplied or allocated HipImage.
 * See warpAffineHip for sampling, borders and initialized transparent output.
 */
CV_EXPORTS_AS(warpAffine) Ptr<HipImage> warpAffineForPython(const Ptr<HipImage>& src, InputArray M,
    Size dsize, Ptr<HipImage> dst = Ptr<HipImage>(), int flags = INTER_LINEAR,
    int borderMode = BORDER_CONSTANT, const Scalar& borderValue = Scalar(),
    Ptr<HipStream> stream = Ptr<HipStream>());

//! @}
}} // namespace cv::triton

#endif // OPENCV_TRITON_HIP_HPP
