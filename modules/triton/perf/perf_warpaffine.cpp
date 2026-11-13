// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Steady-state timing of cv::triton::warpAffine through the public entry, with the native CUDA
// path as a same-card timing baseline.
//
// Unlike resize but like cvtColor, the two sides are close to the same computation: this module
// computes what the CPU float invoker computes, and cv::cuda::warpAffine's textured path is a
// different but comparable resampling. A ratio here is a same-card comparison of two warp
// implementations, NOT a claim about the CPU result -- the module matches the CPU, cv::cuda does
// not (see doc/warpaffine.markdown).
//
// What a ratio does and does not mean at these sizes. The submission floor measured on this
// class of card is tens of microseconds; a warp of a 720p image finishes well under it, so the
// small-image rows are there to show the floor being hit, not to rank the implementations. The
// 1080p and 4K rows are the ones where device work dominates. This is also why the module's own
// banked measurement used graph replay and reported intervals: an event-timed loop at the floor
// measures submission, not the kernel.
//
// Compile and first-load costs are excluded by warming up before the measured cycle, and the
// destination is preallocated so allocation is not in the timed region.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

DEF_PARAM_TEST(Sz_Interp_Dtype, cv::Size, int, int, int);

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

    if (!cv::triton::haveBackend())
        throw SkipTestException("no Triton backend available");

    const int type = CV_MAKETYPE(depth, 3);
    cv::Mat host(size, type);
    declare.in(host, WARMUP_RNG);
    const cv::Mat M = toMat(kGeoms[geom].M);

    const cv::cuda::GpuMat src(host);
    cv::cuda::GpuMat dst(size, type);   // preallocated: reused, excluded from the cycle

    for (int i = 0; i < 20; ++i)
        cv::triton::warpAffine(src, dst, M, size, GET_PARAM(3), cv::BORDER_REPLICATE);
    cv::cuda::Stream::Null().waitForCompletion();

    TEST_CYCLE()
    {
        cv::triton::warpAffine(src, dst, M, size, GET_PARAM(3), cv::BORDER_REPLICATE);
        cv::cuda::Stream::Null().waitForCompletion();
    }

    CUDA_SANITY_CHECK(dst, 1e-10);
}

// AREA aliases LINEAR. Transparent reads existing dst, so these routes have their
// own rows rather than implying an equivalent native CUDA comparison. The initial
// background is defined before warmup; subsequent calls reuse its current contents.
DEF_PARAM_TEST(Sz_Interp_Border_Type, cv::Size, int, int, int);

PERF_TEST_P(Sz_Interp_Border_Type, TritonWarpAffineAreaTransparent, testing::Combine(
    testing::Values(cv::Size(1920, 1080), cv::Size(3840, 2160)),
    testing::Values((int)cv::INTER_NEAREST, (int)cv::INTER_LINEAR, (int)cv::INTER_AREA),
    testing::Values((int)cv::BORDER_REPLICATE, (int)cv::BORDER_TRANSPARENT),
    testing::Values(CV_8UC3, CV_16UC1, CV_32FC1)))
{
    const cv::Size size = GET_PARAM(0);
    const int interp = GET_PARAM(1), border = GET_PARAM(2), type = GET_PARAM(3);
    if (!cv::triton::haveBackend())
        throw SkipTestException("no Triton backend available");

    cv::Mat host(size, type);
    declare.in(host, WARMUP_RNG);
    const cv::Mat M = toMat(kGeoms[2].M);
    const cv::cuda::GpuMat src(host);
    cv::cuda::GpuMat dst(host);
    for (int i = 0; i < 20; ++i)
        cv::triton::warpAffine(src, dst, M, size, interp, border);
    cv::cuda::Stream::Null().waitForCompletion();

    TEST_CYCLE()
    {
        cv::triton::warpAffine(src, dst, M, size, interp, border);
        cv::cuda::Stream::Null().waitForCompletion();
    }
    CUDA_SANITY_CHECK(dst, 1e-10);
}

#ifdef OPENCV_TRITON_PERF_HAVE_CUDAWARPING
// The native implementation on the same card and harness. cudawarping is an OPTIONAL dependency
// of perf/ only -- the module never uses it -- so a build without it omits this baseline.
//
// The bridge is not involved: this is cv::cuda::warpAffine called directly, with the source the
// harness built. It is the same call the module's own baseline used, minus the pitch-staging
// workaround that the lab bridge needed for a non-aligned tensor.
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
