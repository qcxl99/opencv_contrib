// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Steady-state timing of cv::triton::cvtColor through the runtime JIT, with the native CUDA
// path as a same-card timing baseline.
//
// The selected CV_8U conversions use CPU-aligned semantics. The optional cv::cuda baseline
// is identified separately; its presence does not establish equality for every conversion.
// Measurements below include public submission and stream completion, not pure device time.
//
// Compile and first-load costs are excluded by warming up before the measured cycle, and the
// destination is preallocated so allocation is not in the timed region.
//
// One caveat on what a ratio means at these sizes: the submission floor measured on this
// class of card is ~35 us. A conversion that finishes well under that is measured as
// submission-limited, not as the kernel's cost, so the small-image rows are there to show the
// floor being hit rather than to rank the implementations.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef ::perf::TestBaseWithParam<testing::tuple<cv::Size, int>> Sz_Code;

// Seven selected codes with distinct cost structures; this is not coverage of every
// supported code/depth. All candidate paths use the same private validation adapter on
// each backend; the optional native reference below is specifically CUDA.
const int kCodes[] = {
    cv::COLOR_BGR2GRAY,
    cv::COLOR_BGRA2GRAY,
    cv::COLOR_BGR2YUV,
    cv::COLOR_BGR2YCrCb,
    cv::COLOR_BGR2HSV,
    cv::COLOR_BGR2BGRA,
    cv::COLOR_BGRA2BGR,
};

static int scnOf(int code)
{
    return code == cv::COLOR_BGRA2BGR || code == cv::COLOR_BGRA2GRAY ? 4 : 3;
}

static int dcnOf(int code)
{
    return code == cv::COLOR_BGR2BGRA ? 4 : code == cv::COLOR_BGR2GRAY ||
                   code == cv::COLOR_BGRA2GRAY ? 1 : 3;
}

PERF_TEST_P(Sz_Code, TritonCvtColor, testing::Combine(
    testing::Values(cv::Size(640, 480), cv::Size(1920, 1080), cv::Size(3840, 2160)),
    testing::ValuesIn(kCodes)))
{
    const cv::Size size = GET_PARAM(0);
    const int code      = GET_PARAM(1);

    if (!triton_validation::available())
        throw SkipTestException("no Triton backend available");

    cv::Mat host(size, CV_8UC(scnOf(code)));
    declare.in(host, WARMUP_RNG);

    triton_validation::Image src; src.upload(host);
    triton_validation::Stream stream(true);
    triton_validation::Image dst(size, CV_8UC(dcnOf(code)));   // preallocated: reusable, excluded

    for (int i = 0; i < 20; ++i)
        triton_validation::cvtColor(src, dst, code, stream);
    stream.wait();

    TEST_CYCLE()
    {
        triton_validation::cvtColor(src, dst, code, stream);
        stream.wait();
    }

    const cv::Mat result = dst.download();
    SANITY_CHECK(result, 1e-10);
}

#ifdef OPENCV_TRITON_PERF_HAVE_CUDAIMGPROC
// The native implementation on the same card and harness. cudaimgproc is an OPTIONAL
// dependency of perf/ only -- the module itself never uses it -- so a build without it simply
// omits this baseline. The codes restricted to what cv::cuda::cvtColor actually serves: it
// has no BGRA2GRAY and no pure channel permute in the CV_8U path, so those rows are absent
// rather than reported as a failure.
PERF_TEST_P(Sz_Code, NativeCudaCvtColor_Reference, testing::Combine(
    testing::Values(cv::Size(640, 480), cv::Size(1920, 1080), cv::Size(3840, 2160)),
    testing::Values((int)cv::COLOR_BGR2GRAY, (int)cv::COLOR_BGR2YUV,
                    (int)cv::COLOR_BGR2YCrCb, (int)cv::COLOR_BGR2HSV)))
{
    const cv::Size size = GET_PARAM(0);
    const int code      = GET_PARAM(1);

    cv::Mat host(size, CV_8UC(3));
    declare.in(host, WARMUP_RNG);

    const cv::cuda::GpuMat src(host);
    cv::cuda::GpuMat dst(size, CV_8UC(code == cv::COLOR_BGR2GRAY ? 1 : 3));

    for (int i = 0; i < 20; ++i)
        cv::cuda::cvtColor(src, dst, code);
    cv::cuda::Stream::Null().waitForCompletion();

    TEST_CYCLE()
    {
        cv::cuda::cvtColor(src, dst, code);
        cv::cuda::Stream::Null().waitForCompletion();
    }

    CUDA_SANITY_CHECK(dst, 1e-10);
}
#endif  // OPENCV_TRITON_PERF_HAVE_CUDAIMGPROC

}} // namespace
