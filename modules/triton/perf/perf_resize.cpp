// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Warmed public-call timing, including submission and stream completion, not isolated
// kernel or graph time. Compilation, first load and reusable output allocation precede
// the timed cycle. Native CUDA comparison rows are optional and absent on HIP.
// cv::cuda::resize can use different sampling/rounding from this module's CPU reference;
// a timing comparison therefore does not establish numerical equivalence.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef ::perf::TestBaseWithParam<testing::tuple<cv::Size, cv::Size, int, int, int>> Sz_Dsize_Cn_Interp;

PERF_TEST_P(Sz_Dsize_Cn_Interp, TritonResize, testing::Combine(
    testing::Values(cv::Size(1280, 720), cv::Size(1920, 1080), cv::Size(3840, 2160)),
    testing::Values(cv::Size(640, 360), cv::Size(1280, 720), cv::Size(2560, 1440)),
    testing::Values(1, 3, 4),
    testing::Values((int)cv::INTER_NEAREST, (int)cv::INTER_LINEAR, (int)cv::INTER_CUBIC, (int)cv::INTER_AREA),
    testing::Values(CV_8U, CV_32F)))
{
    const cv::Size srcSize = GET_PARAM(0);
    const cv::Size dsize   = GET_PARAM(1);
    const int channels     = GET_PARAM(2);
    const int interp       = GET_PARAM(3);

    if (!triton_validation::available())
        throw SkipTestException("no Triton backend available");

    cv::Mat host(srcSize, CV_MAKETYPE(GET_PARAM(4), channels));
    declare.in(host, WARMUP_RNG);

    triton_validation::Image src; src.upload(host);
    triton_validation::Stream stream(true);
    triton_validation::Image dst(dsize, src.type());   // preallocated: reusable, excluded

    // Warm up compilation/module load and first-touch before measuring.
    for (int i = 0; i < 20; ++i)
        triton_validation::resize(src, dst, dsize, 0, 0, interp, stream);
    stream.wait();

    TEST_CYCLE()
    {
        triton_validation::resize(src, dst, dsize, 0, 0, interp, stream);
        stream.wait();
    }

    const cv::Mat result = dst.download();
    SANITY_CHECK(result, 1e-10);
}

#ifdef OPENCV_TRITON_PERF_HAVE_CUDAWARPING
// Same geometry through the native implementation, so the comparison is on
// one card under one harness rather than across tools. Absent when the build
// has no cudawarping: this is a baseline, not part of the production implementation.
PERF_TEST_P(Sz_Dsize_Cn_Interp, NativeCudaResize_Reference, testing::Combine(
    testing::Values(cv::Size(1280, 720), cv::Size(1920, 1080), cv::Size(3840, 2160)),
    testing::Values(cv::Size(640, 360), cv::Size(1280, 720), cv::Size(2560, 1440)),
    testing::Values(1, 3, 4),
    testing::Values((int)cv::INTER_NEAREST, (int)cv::INTER_LINEAR, (int)cv::INTER_CUBIC, (int)cv::INTER_AREA),
    testing::Values(CV_8U, CV_32F)))
{
    const cv::Size srcSize = GET_PARAM(0);
    const cv::Size dsize   = GET_PARAM(1);
    const int channels     = GET_PARAM(2);
    const int interp       = GET_PARAM(3);

    cv::Mat host(srcSize, CV_MAKETYPE(GET_PARAM(4), channels));
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
#endif  // OPENCV_TRITON_PERF_HAVE_CUDAWARPING

// All four modes are exercised, including fractional AREA downsampling.

}} // namespace
