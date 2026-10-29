// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Steady-state timing of the full operator, with the native CUDA path as an
// explicit same-card comparison. Build and first-load costs are excluded by
// warming up before the measured cycle.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

DEF_PARAM_TEST(Sz_Dsize_Cn_Interp, cv::Size, cv::Size, int, int);
DEF_PARAM_TEST(Sz_Dsize_Cn_Interp_Depth, cv::Size, cv::Size, int, int, int);

PERF_TEST_P(Sz_Dsize_Cn_Interp, TritonResize, testing::Combine(
    testing::Values(cv::Size(1280, 720), cv::Size(1920, 1080), cv::Size(3840, 2160)),
    testing::Values(cv::Size(640, 360), cv::Size(1280, 720), cv::Size(2560, 1440)),
    testing::Values(1, 3),
    testing::Values((int)cv::INTER_NEAREST, (int)cv::INTER_LINEAR)))
{
    const cv::Size srcSize = GET_PARAM(0);
    const cv::Size dsize   = GET_PARAM(1);
    const int channels     = GET_PARAM(2);
    const int interp       = GET_PARAM(3);

    if (!cv::triton::haveBackend())
        throw SkipTestException("no Triton backend available");

    cv::Mat host(srcSize, CV_8UC(channels));
    declare.in(host, WARMUP_RNG);

    const cv::cuda::GpuMat src(host);
    cv::cuda::GpuMat dst(dsize, src.type());   // preallocated: reusable, excluded

    // Warm up compilation/module load and first-touch before measuring.
    for (int i = 0; i < 20; ++i)
        cv::triton::resize(src, dst, dsize, 0, 0, interp);
    cv::cuda::Stream::Null().waitForCompletion();

    TEST_CYCLE()
    {
        cv::triton::resize(src, dst, dsize, 0, 0, interp);
        cv::cuda::Stream::Null().waitForCompletion();
    }

    CUDA_SANITY_CHECK(dst, 1e-10);
}

// Same geometry through the native implementation, so the comparison is on
// one card under one harness rather than across tools.
PERF_TEST_P(Sz_Dsize_Cn_Interp, NativeCudaResize_Reference, testing::Combine(
    testing::Values(cv::Size(1280, 720), cv::Size(1920, 1080), cv::Size(3840, 2160)),
    testing::Values(cv::Size(640, 360), cv::Size(1280, 720), cv::Size(2560, 1440)),
    testing::Values(1, 3),
    testing::Values((int)cv::INTER_NEAREST, (int)cv::INTER_LINEAR)))
{
    const cv::Size srcSize = GET_PARAM(0);
    const cv::Size dsize   = GET_PARAM(1);
    const int channels     = GET_PARAM(2);
    const int interp       = GET_PARAM(3);

    cv::Mat host(srcSize, CV_8UC(channels));
    declare.in(host, WARMUP_RNG);

    const cv::cuda::GpuMat src(host);
    cv::cuda::GpuMat dst(dsize, src.type());

    for (int i = 0; i < 20; ++i)
        cv::cuda::resize(src, dst, dsize, 0, 0, interp);
    cv::cuda::Stream::Null().waitForCompletion();

    TEST_CYCLE()
    {
        cv::cuda::resize(src, dst, dsize, 0, 0, interp);
        cv::cuda::Stream::Null().waitForCompletion();
    }

    CUDA_SANITY_CHECK(dst, 1e-10);
}

// CUBIC and C4 on a separate fixture rather than widening the Values() lists
// above: CUBIC's 3x3 (Sz x Dsize) x 3 (Cn, now including 4) sweep plus a
// dtype axis would multiply the existing NEAREST/LINEAR matrix size for a
// mode whose cost profile (4x4 taps vs LINEAR's 2x2) is not comparable
// case-for-case anyway. Depth is a real perf axis here, unlike above: the
// LINEAR/NEAREST fixtures never varied dtype, so CV_8U was implicit; CUBIC
// adds CV_32F so this fixture's own regressions are visible without
// conflating them with the untouched LINEAR/NEAREST baseline.
PERF_TEST_P(Sz_Dsize_Cn_Interp_Depth, TritonResizeCubic, testing::Combine(
    testing::Values(cv::Size(1280, 720), cv::Size(1920, 1080), cv::Size(3840, 2160)),
    testing::Values(cv::Size(640, 360), cv::Size(1280, 720), cv::Size(2560, 1440)),
    testing::Values(1, 3, 4),
    testing::Values((int)cv::INTER_CUBIC),
    testing::Values((int)CV_8U, (int)CV_32F)))
{
    const cv::Size srcSize = GET_PARAM(0);
    const cv::Size dsize   = GET_PARAM(1);
    const int channels     = GET_PARAM(2);
    const int interp       = GET_PARAM(3);
    const int depth        = GET_PARAM(4);

    if (!cv::triton::haveBackend())
        throw SkipTestException("no Triton backend available");

    cv::Mat host(srcSize, CV_MAKETYPE(depth, channels));
    declare.in(host, WARMUP_RNG);

    const cv::cuda::GpuMat src(host);
    cv::cuda::GpuMat dst(dsize, src.type());

    for (int i = 0; i < 20; ++i)
        cv::triton::resize(src, dst, dsize, 0, 0, interp);
    cv::cuda::Stream::Null().waitForCompletion();

    TEST_CYCLE()
    {
        cv::triton::resize(src, dst, dsize, 0, 0, interp);
        cv::cuda::Stream::Null().waitForCompletion();
    }

    CUDA_SANITY_CHECK(dst, 1e-10);
}

PERF_TEST_P(Sz_Dsize_Cn_Interp_Depth, NativeCudaResizeCubic_Reference, testing::Combine(
    testing::Values(cv::Size(1280, 720), cv::Size(1920, 1080), cv::Size(3840, 2160)),
    testing::Values(cv::Size(640, 360), cv::Size(1280, 720), cv::Size(2560, 1440)),
    testing::Values(1, 3, 4),
    testing::Values((int)cv::INTER_CUBIC),
    testing::Values((int)CV_8U, (int)CV_32F)))
{
    const cv::Size srcSize = GET_PARAM(0);
    const cv::Size dsize   = GET_PARAM(1);
    const int channels     = GET_PARAM(2);
    const int interp       = GET_PARAM(3);
    const int depth        = GET_PARAM(4);

    cv::Mat host(srcSize, CV_MAKETYPE(depth, channels));
    declare.in(host, WARMUP_RNG);

    const cv::cuda::GpuMat src(host);
    cv::cuda::GpuMat dst(dsize, src.type());

    for (int i = 0; i < 20; ++i)
        cv::cuda::resize(src, dst, dsize, 0, 0, interp);
    cv::cuda::Stream::Null().waitForCompletion();

    TEST_CYCLE()
    {
        cv::cuda::resize(src, dst, dsize, 0, 0, interp);
        cv::cuda::Stream::Null().waitForCompletion();
    }

    CUDA_SANITY_CHECK(dst, 1e-10);
}

// AREA on its own fixture, same reasoning as CUBIC above: a cost profile
// (two different compiled formulas, IntegerAreaFilter vs AreaFilter,
// selected by whether the ratio is an exact integer) not comparable
// case-for-case to LINEAR/NEAREST/CUBIC's fixtures. Downscale-only
// geometries, deliberately: an upscale-or-identity-on-either-axis AREA
// call aliases to LINEAR semantics before this module's backend ever
// sees it (resize.cu:736-737), so an upscale case here would just
// re-measure the LINEAR fixture above under a different name. The two
// Dsize values are chosen to land on each of AREA's two branches at a
// realistic ratio: 1280x720 is an exact 2x/3x-ish integer ratio from each
// Sz value (IntegerAreaFilter), 853x480 is not (AreaFilter).
PERF_TEST_P(Sz_Dsize_Cn_Interp, TritonResizeArea, testing::Combine(
    testing::Values(cv::Size(1280, 720), cv::Size(1920, 1080), cv::Size(3840, 2160)),
    testing::Values(cv::Size(640, 360), cv::Size(853, 480)),
    testing::Values(1, 3, 4),
    testing::Values((int)cv::INTER_AREA)))
{
    const cv::Size srcSize = GET_PARAM(0);
    const cv::Size dsize   = GET_PARAM(1);
    const int channels     = GET_PARAM(2);
    const int interp       = GET_PARAM(3);

    if (!cv::triton::haveBackend())
        throw SkipTestException("no Triton backend available");

    cv::Mat host(srcSize, CV_8UC(channels));
    declare.in(host, WARMUP_RNG);

    const cv::cuda::GpuMat src(host);
    cv::cuda::GpuMat dst(dsize, src.type());

    for (int i = 0; i < 20; ++i)
        cv::triton::resize(src, dst, dsize, 0, 0, interp);
    cv::cuda::Stream::Null().waitForCompletion();

    TEST_CYCLE()
    {
        cv::triton::resize(src, dst, dsize, 0, 0, interp);
        cv::cuda::Stream::Null().waitForCompletion();
    }

    CUDA_SANITY_CHECK(dst, 1e-10);
}

PERF_TEST_P(Sz_Dsize_Cn_Interp, NativeCudaResizeArea_Reference, testing::Combine(
    testing::Values(cv::Size(1280, 720), cv::Size(1920, 1080), cv::Size(3840, 2160)),
    testing::Values(cv::Size(640, 360), cv::Size(853, 480)),
    testing::Values(1, 3, 4),
    testing::Values((int)cv::INTER_AREA)))
{
    const cv::Size srcSize = GET_PARAM(0);
    const cv::Size dsize   = GET_PARAM(1);
    const int channels     = GET_PARAM(2);
    const int interp       = GET_PARAM(3);

    cv::Mat host(srcSize, CV_8UC(channels));
    declare.in(host, WARMUP_RNG);

    const cv::cuda::GpuMat src(host);
    cv::cuda::GpuMat dst(dsize, src.type());

    for (int i = 0; i < 20; ++i)
        cv::cuda::resize(src, dst, dsize, 0, 0, interp);
    cv::cuda::Stream::Null().waitForCompletion();

    TEST_CYCLE()
    {
        cv::cuda::resize(src, dst, dsize, 0, 0, interp);
        cv::cuda::Stream::Null().waitForCompletion();
    }

    CUDA_SANITY_CHECK(dst, 1e-10);
}

}}  // namespace opencv_test::<anonymous>
