// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Warmed public-call timing with optional same-device native CUDA comparisons.
// The selected CPU reference and cv::cuda::warpAffine use different sampling
// arithmetic; timing ratios do not imply numerical equivalence.
// Compilation, first load and output allocation occur before the timed cycle.
// Public submission and stream completion are included. Host overhead can
// dominate short operations; these rows are not isolated kernel timings.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef ::perf::TestBaseWithParam<testing::tuple<cv::Size, int, int, int>> Sz_Interp_Dtype;

// Three geometries, chosen for different cost structures rather than to be exhaustive:
// identity-translate (memory-bandwidth bound, the cheapest mapping), a half-pixel translate
// (every LINEAR tap is fractional, the most expensive weight path), and a rotate+scale (the
// gather pattern is non-monotonic, so the load stream is not sequential).
struct WaGeometry
{
    const char* name;
    double M[6];
};

static const WaGeometry kGeoms[] = {
    {"identity",  {1.0, 0.0, 0.0, 0.0, 1.0, 0.0}},
    {"subpixel",  {1.0, 0.0, 0.5, 0.0, 1.0, 0.5}},
    {"rot_scale", {0.9, -0.43589, 40.0, 0.43589, 0.9, -20.0}},
};

static cv::Mat toMat(const double m[6])
{
    cv::Mat M(2, 3, CV_64F);
    for (int i = 0; i < 6; ++i)
        M.at<double>(i / 3, i % 3) = m[i];
    return M;
}

// Typed so a 16-bit row is visible in the output name; the depth is one value per run rather
// than a range, because a perf row should answer one question.
PERF_TEST_P(Sz_Interp_Dtype, TritonWarpAffine, testing::Combine(
    testing::Values(cv::Size(1280, 720), cv::Size(1920, 1080), cv::Size(3840, 2160)),
    testing::Values(0, 1, 2),          // index into kGeoms, so the name is visible in output
    testing::Values(CV_8U, CV_16U, CV_32F),
    testing::Values((int)cv::INTER_NEAREST, (int)cv::INTER_LINEAR, (int)cv::INTER_CUBIC)))
{
    const cv::Size size = GET_PARAM(0);
    const int geom      = GET_PARAM(1);
    const int depth     = GET_PARAM(2);

    if (!triton_validation::available())
        throw SkipTestException("no Triton backend available");

    const int type = CV_MAKETYPE(depth, 3);
    cv::Mat host(size, type);
    declare.in(host, WARMUP_RNG);
    const cv::Mat M = toMat(kGeoms[geom].M);

    triton_validation::Image src; src.upload(host);
    triton_validation::Stream stream(true);
    triton_validation::Image dst(size, type);   // preallocated: reused, excluded from the cycle

    for (int i = 0; i < 20; ++i)
        triton_validation::warpAffine(src, dst, M, size, GET_PARAM(3), cv::BORDER_REPLICATE, cv::Scalar(), stream);
    stream.wait();

    TEST_CYCLE()
    {
        triton_validation::warpAffine(src, dst, M, size, GET_PARAM(3), cv::BORDER_REPLICATE, cv::Scalar(), stream);
        stream.wait();
    }

    const cv::Mat result = dst.download();
    SANITY_CHECK(result, 1e-10);
}

// AREA aliases LINEAR. Transparent reads existing dst, so these routes have their
// own rows rather than implying an equivalent native CUDA comparison. The initial
// background is defined before warmup; subsequent calls reuse its current contents.
typedef ::perf::TestBaseWithParam<testing::tuple<cv::Size, int, int, int>> Sz_Interp_Border_Type;

PERF_TEST_P(Sz_Interp_Border_Type, TritonWarpAffineAreaTransparent, testing::Combine(
    testing::Values(cv::Size(1920, 1080), cv::Size(3840, 2160)),
    testing::Values((int)cv::INTER_NEAREST, (int)cv::INTER_LINEAR, (int)cv::INTER_AREA),
    testing::Values((int)cv::BORDER_REPLICATE, (int)cv::BORDER_TRANSPARENT),
    testing::Values(CV_8UC3, CV_16UC1, CV_32FC1)))
{
    const cv::Size size = GET_PARAM(0);
    const int interp = GET_PARAM(1), border = GET_PARAM(2), type = GET_PARAM(3);
    if (!triton_validation::available())
        throw SkipTestException("no Triton backend available");

    cv::Mat host(size, type);
    declare.in(host, WARMUP_RNG);
    const cv::Mat M = toMat(kGeoms[2].M);
    triton_validation::Image src; src.upload(host);
    triton_validation::Stream stream(true);
    triton_validation::Image dst; dst.upload(host);
    for (int i = 0; i < 20; ++i)
        triton_validation::warpAffine(src, dst, M, size, interp, border, cv::Scalar(), stream);
    stream.wait();

    TEST_CYCLE()
    {
        triton_validation::warpAffine(src, dst, M, size, interp, border, cv::Scalar(), stream);
        stream.wait();
    }
    const cv::Mat result = dst.download();
    SANITY_CHECK(result, 1e-10);
}

#ifdef OPENCV_TRITON_PERF_HAVE_CUDAWARPING
// The native implementation on the same card and harness. cudawarping is an OPTIONAL dependency
// of perf/ only -- the module never uses it -- so a build without it omits this baseline.
//
// Call the native cv::cuda entry directly with the harness-owned GpuMat.
PERF_TEST_P(Sz_Interp_Dtype, NativeCudaWarpAffine_Reference, testing::Combine(
    testing::Values(cv::Size(1280, 720), cv::Size(1920, 1080), cv::Size(3840, 2160)),
    testing::Values(0, 1, 2),
    testing::Values(CV_8U, CV_16U, CV_32F),
    testing::Values((int)cv::INTER_NEAREST, (int)cv::INTER_LINEAR, (int)cv::INTER_CUBIC)))
{
    const cv::Size size = GET_PARAM(0);
    const int geom      = GET_PARAM(1);
    const int depth     = GET_PARAM(2);

    const int type = CV_MAKETYPE(depth, 3);
    cv::Mat host(size, type);
    declare.in(host, WARMUP_RNG);
    const cv::Mat M = toMat(kGeoms[geom].M);

    const cv::cuda::GpuMat src(host);
    cv::cuda::GpuMat dst(size, type);

    for (int i = 0; i < 20; ++i)
        cv::cuda::warpAffine(src, dst, M, size, GET_PARAM(3), cv::BORDER_REPLICATE);
    cv::cuda::Stream::Null().waitForCompletion();

    TEST_CYCLE()
    {
        cv::cuda::warpAffine(src, dst, M, size, GET_PARAM(3), cv::BORDER_REPLICATE);
        cv::cuda::Stream::Null().waitForCompletion();
    }

    CUDA_SANITY_CHECK(dst, 1e-10);
}
#endif  // OPENCV_TRITON_PERF_HAVE_CUDAWARPING

}} // namespace
