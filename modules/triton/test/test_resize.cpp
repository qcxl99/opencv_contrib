// Semantics, numerics, interop and declared error behaviour.
//
// The reference is cv::cuda::resize on the SAME card, because that is what
// this module's contract targets. It is deliberately NOT OpenCV CPU resize:
// for INTER_LINEAR the CUDA rule omits the half-pixel shift, so CPU and CUDA
// disagree by design and a CPU comparison would fail a correct kernel.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv;

static Mat makePattern(int rows, int cols, int type, int kind)
{
    Mat m(rows, cols, type);
    const int cn = CV_MAT_CN(type);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            for (int c = 0; c < cn; ++c)
            {
                uchar v = 0;
                switch (kind)
                {
                case 0: v = static_cast<uchar>((y * 7 + x * 13 + c * 29) & 0xFF); break; // pseudo-random
                case 1: v = static_cast<uchar>(((y + x) % 256)); break;                 // gradient
                case 2: v = static_cast<uchar>(((y + x) % 2) ? 255 : 0); break;          // checkerboard
                case 3: v = static_cast<uchar>(x % 256); break;                          // coordinate
                default: v = 128; break;                                                 // constant
                }
                m.at<Vec<uchar, 1> >(y, x * cn + c)[0] = v;
            }
    return m;
}

/** A CUDA context must exist before the backend can report availability --
 *  cuCtxGetCurrent() returns null until the runtime has been touched. Probing
 *  haveBackend() on a cold process therefore reports false and would silently
 *  skip every case, which is exactly what happened the first time this suite
 *  ran (4800/4800 skipped). Establish the context first. */
static bool tritonReady()
{
    static const bool ready = []() {
        if (cv::cuda::getCudaEnabledDeviceCount() < 1)
            return false;
        cuda::GpuMat warm(4, 4, CV_8UC1);
        warm.setTo(Scalar::all(0));
        cuda::Stream::Null().waitForCompletion();
        return cv::triton::haveBackend();
    }();
    return ready;
}

struct TritonResize : testing::TestWithParam<testing::tuple<Size, Size, int, int, int> > {};

CUDA_TEST_P(TritonResize, MatchesNativeCuda)
{
    if (!tritonReady())
        throw SkipTestException("no Triton backend available");

    const Size srcSize = testing::get<0>(GetParam());
    const Size dsize   = testing::get<1>(GetParam());
    const int channels = testing::get<2>(GetParam());
    const int interp   = testing::get<3>(GetParam());
    const int pattern  = testing::get<4>(GetParam());

    const int type = CV_8UC(channels);
    const Mat host = makePattern(srcSize.height, srcSize.width, type, pattern);

    cuda::GpuMat src(host), expected, actual;
    cuda::resize(src, expected, dsize, 0, 0, interp);
    cv::triton::resize(src, actual, dsize, 0, 0, interp);
    cuda::Stream::Null().waitForCompletion();

    ASSERT_EQ(expected.size(), actual.size());
    ASSERT_EQ(expected.type(), actual.type());
    // Bit-exact: the contract is exact agreement with the native CUDA kernel.
    EXPECT_MAT_NEAR(Mat(expected), Mat(actual), 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Triton, TritonResize, testing::Combine(
    testing::Values(Size(1, 1), Size(1, 64), Size(64, 1), Size(2, 2),
                    Size(17, 33), Size(64, 64), Size(640, 480), Size(1920, 1080)),
    testing::Values(Size(1, 1), Size(8, 8), Size(32, 32), Size(34, 34),
                    Size(128, 128), Size(1281, 721)),
    testing::Values(1, 3),
    testing::Values((int)INTER_NEAREST, (int)INTER_LINEAR),
    testing::Values(0, 1, 2, 3, 4)));

// --- interop ------------------------------------------------------------

CUDA_TEST_P(TritonResize, ReusesOutputAndLeavesInputIntact)
{
    if (!tritonReady())
        throw SkipTestException("no Triton backend available");

    const Mat host = makePattern(480, 640, CV_8UC3, 0);
    cuda::GpuMat src(host);
    cuda::GpuMat dst(240, 320, CV_8UC3);
    const void* addr = dst.cudaPtr();

    for (int i = 0; i < 3; ++i)
        cv::triton::resize(src, dst, Size(320, 240), 0, 0, INTER_LINEAR);
    cuda::Stream::Null().waitForCompletion();

    EXPECT_EQ(addr, dst.cudaPtr()) << "preallocated output must not be reallocated";

    Mat back;
    src.download(back);
    EXPECT_EQ(0, cv::norm(host, back, NORM_INF)) << "input must not be modified";
}

CUDA_TEST_P(TritonResize, NonDefaultStreamMatches)
{
    if (!tritonReady())
        throw SkipTestException("no Triton backend available");

    const Mat host = makePattern(480, 640, CV_8UC1, 1);
    cuda::GpuMat src(host), a, b;
    cuda::Stream stream;

    cv::triton::resize(src, a, Size(320, 240), 0, 0, INTER_LINEAR);
    cuda::Stream::Null().waitForCompletion();
    cv::triton::resize(src, b, Size(320, 240), 0, 0, INTER_LINEAR, stream);
    stream.waitForCompletion();

    EXPECT_MAT_NEAR(Mat(a), Mat(b), 0.0);
}

CUDA_TEST_P(TritonResize, OutputIsReallocatedWhenNeeded)
{
    if (!tritonReady())
        throw SkipTestException("no Triton backend available");

    const Mat host = makePattern(64, 64, CV_8UC1, 0);
    cuda::GpuMat src(host);
    cuda::GpuMat dst;   // empty: create() must size it
    cv::triton::resize(src, dst, Size(32, 32), 0, 0, INTER_NEAREST);
    cuda::Stream::Null().waitForCompletion();
    EXPECT_EQ(Size(32, 32), dst.size());
    EXPECT_EQ(CV_8UC1, dst.type());
}

// --- declared error behaviour -------------------------------------------

CUDA_TEST_P(TritonResize, RejectsUnsupportedInputs)
{
    if (!tritonReady())
        throw SkipTestException("no Triton backend available");

    cuda::GpuMat u8c1(64, 64, CV_8UC1, Scalar::all(0));
    cuda::GpuMat dst;

    // float depth
    {
        cuda::GpuMat f32(64, 64, CV_32FC1, Scalar::all(0));
        EXPECT_THROW(cv::triton::resize(f32, dst, Size(32, 32)), cv::Exception);
    }
    // 4 channels
    {
        cuda::GpuMat u8c4(64, 64, CV_8UC4, Scalar::all(0));
        EXPECT_THROW(cv::triton::resize(u8c4, dst, Size(32, 32)), cv::Exception);
    }
    // unsupported interpolation
    EXPECT_THROW(cv::triton::resize(u8c1, dst, Size(32, 32), 0, 0, INTER_CUBIC),
                 cv::Exception);
    // empty dsize with fx/fy -- the scale-factor path is not supported
    EXPECT_THROW(cv::triton::resize(u8c1, dst, Size(), 0.5, 0.5), cv::Exception);
    // in-place
    EXPECT_THROW(cv::triton::resize(u8c1, u8c1, Size(32, 32)), cv::Exception);
    // ROI source
    {
        cuda::GpuMat parent(128, 128, CV_8UC1, Scalar::all(0));
        cuda::GpuMat roi = parent(Rect(8, 8, 64, 64));
        EXPECT_THROW(cv::triton::resize(roi, dst, Size(32, 32)), cv::Exception);
    }
    // ROI DESTINATION. Checked separately because only the source was
    // validated at first: a ROI dst reached the kernel with a pitch the
    // artifact selection had not accounted for.
    {
        cuda::GpuMat parent(128, 128, CV_8UC1, Scalar::all(0));
        cuda::GpuMat roiDst = parent(Rect(8, 8, 32, 32));
        EXPECT_THROW(cv::triton::resize(u8c1, roiDst, Size(32, 32)), cv::Exception);
    }
    // A ROI dst whose size already matches dsize: create() returns it
    // unchanged, so the post-create check is the one that must catch it.
    {
        cuda::GpuMat parent(256, 256, CV_8UC3, Scalar::all(0));
        cuda::GpuMat roiDst = parent(Rect(16, 16, 64, 64));
        cuda::GpuMat src3(128, 128, CV_8UC3, Scalar::all(7));
        EXPECT_THROW(cv::triton::resize(src3, roiDst, Size(64, 64)), cv::Exception);
    }
}

/** haveBackend() must be a pure query: no allocation, no submitted work.
 *
 *  It used to allocate a 1x1 GpuMat and call setTo just to force a context
 *  into existence. Verified by comparing free device memory across repeated
 *  calls, so a reintroduced allocation is caught rather than reviewed for. */
CUDA_TEST_P(TritonResize, HaveBackendAllocatesNothing)
{
    if (!tritonReady())
        throw SkipTestException("no Triton backend available");

    size_t freeBefore = 0, totalBefore = 0;
    cuda::DeviceInfo().queryMemory(freeBefore, totalBefore);

    for (int i = 0; i < 64; ++i)
        EXPECT_TRUE(cv::triton::haveBackend());
    EXPECT_FALSE(cv::triton::backendName().empty());

    size_t freeAfter = 0, totalAfter = 0;
    cuda::DeviceInfo().queryMemory(freeAfter, totalAfter);
    EXPECT_EQ(freeBefore, freeAfter)
        << "haveBackend() changed free device memory; it must not allocate";
}

}}  // namespace opencv_test::<anonymous>
