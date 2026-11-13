// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Steady-state timing of the full operator through the runtime JIT, with the native CUDA path
// as a same-card timing baseline. cv::cuda::resize computes a DIFFERENT result (this module
// follows CPU cv::resize), so a ratio here compares two computations, not two implementations
// of one. Compile and first-load costs are excluded by warming up before the measured cycle.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

DEF_PARAM_TEST(Sz_Dsize_Cn_Interp, cv::Size, cv::Size, int, int, int);

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

    if (!cv::triton::haveBackend())
        throw SkipTestException("no Triton backend available");

    cv::Mat host(srcSize, CV_MAKETYPE(GET_PARAM(4), channels));
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

#ifdef OPENCV_TRITON_PERF_HAVE_CUDAWARPING
// Same geometry through the native implementation, so the comparison is on
// one card under one harness rather than across tools. Absent when the build
// has no cudawarping: this is a baseline, not part of the module's contract.
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
