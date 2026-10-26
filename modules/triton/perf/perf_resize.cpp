// Steady-state timing of the full operator, with the native CUDA path as an
// explicit same-card comparison. Build and first-load costs are excluded by
// warming up before the measured cycle.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

DEF_PARAM_TEST(Sz_Dsize_Cn_Interp, cv::Size, cv::Size, int, int);

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

}}  // namespace opencv_test::<anonymous>
