// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Shared by test_resize.cpp and test_runtime.cpp.
#ifndef OPENCV_TRITON_TEST_HELPERS_HPP
#define OPENCV_TRITON_TEST_HELPERS_HPP

#include "test_precomp.hpp"
#include "../src/runtime/reference_cpu.hpp"
#include "../src/runtime/runtime.hpp"

namespace opencv_test { namespace triton_test {

namespace ref = kf_resize_ref;
namespace td = cv::triton::detail;

/** A CUDA context must exist before anything device-side runs. */
inline bool ready()
{
    static const bool ok = []() {
        if (cv::cuda::getCudaEnabledDeviceCount() < 1)
            return false;
        cuda::GpuMat warm(4, 4, CV_8UC1);
        warm.setTo(Scalar::all(0));
        cuda::Stream::Null().waitForCompletion();
        return cv::triton::haveBackend();
    }();
    return ok;
}

inline Mat randomImage(Size sz, int type, uint64 seed)
{
    RNG rng(seed);
    Mat m(sz, type);
    if (CV_MAT_DEPTH(type) == CV_8U)
        rng.fill(m, RNG::UNIFORM, 0, 256);
    else if (CV_MAT_DEPTH(type) == CV_16U)
        // The full 16-bit range: a fill of 0..2 would pass every test while exercising none
        // of the range where a saturating store can go wrong.
        rng.fill(m, RNG::UNIFORM, 0, 65536);
    else
        rng.fill(m, RNG::UNIFORM, -2.0, 2.0);
    return m;
}

/** cvtColor's reference: this build's own cv::cvtColor -- and that is stated as such.
 *
 * For resize the host build is NOT a valid reference (the CPU result depends on the build),
 * which is why a portable oracle exists there. For cvtColor the host build IS the reference,
 * but not because the arithmetic is integer: an integer formula computed differently is still
 * a different formula. It is the reference because each served code is integer, so an output
 * pixel depends only on its own bytes and constant tables -- no reduction order, no SIMD width,
 * no position within the row -- and because that equality was measured on the build recorded in
 * ReferenceBuildIsCharacterised. It is NOT established that some other build computes the same
 * numbers: cv::cvtColor dispatches through cv_hal_cvtBGRtoBGR / cvtBGRtoYUV / cvtBGRtoHSV, so a
 * build with IPP or a custom HAL may differ, and the claim is scoped to the configuration the
 * evidence was gathered on. The codes that are not position independent at all (HSV2BGR, HLS --
 * float32) are refused by the module and never reach here.
 */
inline Mat cpuCvt(const Mat& src, int code)
{
    Mat dst;
    cv::cvtColor(src, dst, code);
    return dst;
}

/** The contract's answer: the portable CPU oracle, independent of this build's cv::resize. */
inline Mat oracle(const Mat& src, Size dsize, double fx, double fy, int interp)
{
    CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
    const ref::Size2 z = ref::resolve_size(src.rows, src.cols, dsize.width, dsize.height, fx, fy);
    Mat dst(z.dst_h, z.dst_w, src.type());
    ref::ImageRef s{src.data, src.rows, src.cols, src.channels(), src.step[0], src.depth() == CV_32F};
    ref::ImageRef d{dst.data, dst.rows, dst.cols, dst.channels(), dst.step[0], src.depth() == CV_32F};
    ref::resize(s, d, z.inv_y, z.inv_x, interp);
    return dst;
}

inline double f32Atol(int interp)   // contract cpu_aligned: 2E, derived from CPU's own rounding
{
    return interp == INTER_NEAREST ? 0.0 : (interp == INTER_LINEAR ? 4.948e-07 :
           (interp == INTER_CUBIC ? 1.401e-06 : 8.419e-07));
}

/** Under the contract: CV_8U exact; CV_32F within the derived bound, NaN matched by NaN. */
inline ::testing::AssertionResult meetsContract(const Mat& got, const Mat& ref_, int interp)
{
    if (got.size() != ref_.size() || got.type() != ref_.type())
        return ::testing::AssertionFailure() << "size/type " << got.size() << "/" << got.type() << " vs " << ref_.size() << "/" << ref_.type();
    const int W = got.cols * got.channels();
    long long bad = 0;
    double worst = 0;
    for (int y = 0; y < got.rows; ++y)
    {
        if (got.depth() == CV_8U)
        {
            const uchar* a = got.ptr<uchar>(y); const uchar* b = ref_.ptr<uchar>(y);
            for (int x = 0; x < W; ++x)
                if (a[x] != b[x]) { ++bad; worst = std::max(worst, std::abs(double(a[x]) - b[x])); }
        }
        else
        {
            const float* a = got.ptr<float>(y); const float* b = ref_.ptr<float>(y);
            for (int x = 0; x < W; ++x)
            {
                if (std::isnan(a[x]) && std::isnan(b[x])) continue;
                const double d = std::abs(double(a[x]) - b[x]);
                if (!(d <= f32Atol(interp))) { ++bad; if (!(d <= worst)) worst = d; }
            }
        }
    }
    if (bad)
        return ::testing::AssertionFailure() << bad << " elements outside the contract, worst |diff| " << worst;
    return ::testing::AssertionSuccess();
}

/** Upload into a padded parent and return the ROI view; `parent` keeps the guard bands. */
inline cuda::GpuMat roiOf(cuda::GpuMat& parent, Size sz, int type, Point ofs, int padX, int padY, double guard)
{
    parent.create(sz.height + padY, sz.width + padX, type);
    parent.setTo(Scalar::all(guard));
    return parent(Rect(ofs, sz));
}

/** Everything in `parent` outside `r` still equals `guard`. */
inline bool guardsIntact(const cuda::GpuMat& parent, Rect r, double guard)
{
    Mat m;
    parent.download(m);
    const int cn = m.channels();
    for (int y = 0; y < m.rows; ++y)
        for (int x = 0; x < m.cols; ++x)
        {
            if (r.contains(Point(x, y)))
                continue;
            for (int c = 0; c < cn; ++c)
            {
                const double v = m.depth() == CV_8U ? double(m.ptr<uchar>(y)[x * cn + c]) : double(m.ptr<float>(y)[x * cn + c]);
                if (v != guard)
                    return false;
            }
        }
    return true;
}

}}  // namespace opencv_test::triton_test

#endif
