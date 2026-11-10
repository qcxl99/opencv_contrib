// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// cv::triton::resize through its real public entry (GpuMat + C++ launcher + binaries from the
// runtime JIT) against the portable CPU oracle: the P1 routes copy, nearest, linear, area_fast x
// CV_8U/CV_32F x C1/C3/C4 x whole images and ROI views, with output reuse and guard bands. The
// reference is deliberately not this build's cv::resize, whose result depends on the build, and
// not cv::cuda::resize, which computes something else.

#include "test_helpers.hpp"

namespace opencv_test { namespace {

using namespace triton_test;

struct RouteCase
{
    Size src, dst;
    double fx, fy;
    int interp;
    const char* route;
};

const RouteCase kCases[] = {
    // copy: identity, whatever the mode
    {Size(97, 53), Size(97, 53), 0, 0, INTER_NEAREST, "copy"},
    {Size(97, 53), Size(97, 53), 0, 0, INTER_LINEAR, "copy"},
    {Size(97, 53), Size(97, 53), 0, 0, INTER_AREA, "copy"},
    {Size(97, 53), Size(97, 53), 0, 0, INTER_CUBIC, "copy"},
    // nearest: down, up, exact ratios, one-pixel edges, fx/fy
    {Size(131, 97), Size(61, 53), 0, 0, INTER_NEAREST, "nearest"},
    {Size(64, 48), Size(200, 150), 0, 0, INTER_NEAREST, "nearest"},
    {Size(128, 96), Size(64, 48), 0, 0, INTER_NEAREST, "nearest"},
    {Size(640, 480), Size(1920, 1080), 0, 0, INTER_NEAREST, "nearest"},
    {Size(1, 1), Size(3, 2), 0, 0, INTER_NEAREST, "nearest"},
    {Size(1, 40), Size(5, 20), 0, 0, INTER_NEAREST, "nearest"},
    {Size(40, 1), Size(20, 5), 0, 0, INTER_NEAREST, "nearest"},
    {Size(64, 48), Size(), 1.5, 0.4, INTER_NEAREST, "nearest"},
    // linear: non-integer down, up, exact 2x up, mixed axes, one-pixel edges, fx/fy
    {Size(131, 97), Size(61, 53), 0, 0, INTER_LINEAR, "linear"},
    {Size(64, 48), Size(200, 150), 0, 0, INTER_LINEAR, "linear"},
    {Size(64, 48), Size(128, 96), 0, 0, INTER_LINEAR, "linear"},
    {Size(640, 480), Size(1920, 1080), 0, 0, INTER_LINEAR, "linear"},
    {Size(60, 90), Size(90, 60), 0, 0, INTER_LINEAR, "linear"},
    {Size(1, 1), Size(7, 5), 0, 0, INTER_LINEAR, "linear"},
    {Size(40, 1), Size(60, 3), 0, 0, INTER_LINEAR, "linear"},
    {Size(64, 48), Size(), 0.7, 1.3, INTER_LINEAR, "linear"},
    // AREA with an upscaled axis is the linear path with AREA's coefficient formula
    {Size(64, 48), Size(200, 150), 0, 0, INTER_AREA, "linear"},
    {Size(64, 48), Size(128, 24), 0, 0, INTER_AREA, "linear"},
    // area_fast: exact 2x LINEAR is routed here, integer ratios, anisotropic, odd sources via fx/fy
    {Size(128, 96), Size(64, 48), 0, 0, INTER_LINEAR, "area_fast"},
    {Size(1920, 1080), Size(960, 540), 0, 0, INTER_LINEAR, "area_fast"},
    {Size(128, 96), Size(64, 48), 0, 0, INTER_AREA, "area_fast"},
    {Size(129, 96), Size(43, 32), 0, 0, INTER_AREA, "area_fast"},
    {Size(160, 120), Size(40, 30), 0, 0, INTER_AREA, "area_fast"},
    {Size(640, 480), Size(80, 60), 0, 0, INTER_AREA, "area_fast"},
    {Size(120, 60), Size(40, 30), 0, 0, INTER_AREA, "area_fast"},
    {Size(129, 97), Size(), 0.5, 0.5, INTER_AREA, "area_fast"},
    {Size(129, 97), Size(), 0.5, 0.5, INTER_LINEAR, "area_fast"},
    {Size(131, 134), Size(), 0.25, 0.25, INTER_AREA, "area_fast"},
};

const int kTypes[] = {CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4};

static std::string label(const RouteCase& c, int type, const char* layout)
{
    return cv::format("%dx%d -> %dx%d fx=%g fy=%g interp=%d type=%s %s", c.src.width, c.src.height, c.dst.width, c.dst.height,
                      c.fx, c.fy, c.interp, typeToString(type).c_str(), layout);
}

// Every case on a non-default stream: whole images, new output buffer.
TEST(Triton_Resize, RoutesWholeImagesOnStream)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    uint64 seed = 1;
    cuda::Stream stream;
    for (const RouteCase& c : kCases)
        for (int type : kTypes)
        {
            SCOPED_TRACE(label(c, type, "whole"));
            const Mat host = randomImage(c.src, type, seed++);
            cuda::GpuMat src(host), dst;
            cv::triton::resize(src, dst, c.dst, c.fx, c.fy, c.interp, stream);
            stream.waitForCompletion();
            EXPECT_STREQ(c.route, td::runtimeStats().last_route.c_str());
            Mat got;
            dst.download(got);
            EXPECT_TRUE(meetsContract(got, oracle(host, c.dst, c.fx, c.fy, c.interp), c.interp));
        }
}

// ROI source and destination with padded steps and odd offsets: the output is reused (same
// address), the result equals the oracle, nothing outside the destination ROI is written.
TEST(Triton_Resize, RoutesRoiAndOutputReuse)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    uint64 seed = 100000;
    cuda::Stream stream;
    for (const RouteCase& c : kCases)
        for (int type : kTypes)
        {
            SCOPED_TRACE(label(c, type, "roi"));
            const Mat host = randomImage(c.src, type, seed++);
            const Size dsz = c.dst.empty() ? Size(cvRound(c.src.width * c.fx), cvRound(c.src.height * c.fy)) : c.dst;
            const double guard = CV_MAT_DEPTH(type) == CV_8U ? 77.0 : 77.5;
            cuda::GpuMat srcParent, dstParent;
            const Point so(3, 1), dso(5, 2);
            cuda::GpuMat src = roiOf(srcParent, c.src, type, so, 7, 3, 0.0);
            cuda::GpuMat dst = roiOf(dstParent, dsz, type, dso, 9, 5, guard);
            src.upload(host);
            const void* before = dst.cudaPtr();
            cv::triton::resize(src, dst, c.dst, c.fx, c.fy, c.interp, stream);
            stream.waitForCompletion();
            EXPECT_EQ(before, dst.cudaPtr()) << "the output was reallocated";
            EXPECT_STREQ(c.route, td::runtimeStats().last_route.c_str());
            Mat got;
            dst.download(got);
            EXPECT_TRUE(meetsContract(got, oracle(host, c.dst, c.fx, c.fy, c.interp), c.interp));
            EXPECT_TRUE(guardsIntact(dstParent, Rect(dso, dsz), guard)) << "wrote outside the destination ROI";
        }
}

// The default (null) stream takes the same path.
TEST(Triton_Resize, DefaultStream)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    uint64 seed = 200000;
    for (const RouteCase& c : kCases)
        for (int type : {CV_8UC3, CV_32FC4})
        {
            SCOPED_TRACE(label(c, type, "null stream"));
            const Mat host = randomImage(c.src, type, seed++);
            cuda::GpuMat src(host), dst;
            cv::triton::resize(src, dst, c.dst, c.fx, c.fy, c.interp);
            cuda::Stream::Null().waitForCompletion();
            Mat got;
            dst.download(got);
            EXPECT_TRUE(meetsContract(got, oracle(host, c.dst, c.fx, c.fy, c.interp), c.interp));
        }
}

TEST(Triton_Resize, CopyRouteIsNotALaunch)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    const Mat host = randomImage(Size(33, 21), CV_8UC3, 7);
    cuda::GpuMat src(host), dst;
    const td::RuntimeStats a = td::runtimeStats();
    cv::triton::resize(src, dst, host.size(), 0, 0, INTER_CUBIC);
    cuda::Stream::Null().waitForCompletion();
    const td::RuntimeStats b = td::runtimeStats();
    EXPECT_EQ(a.copies + 1, b.copies);
    EXPECT_EQ(a.launches, b.launches);
    Mat got;
    dst.download(got);
    EXPECT_EQ(0.0, cvtest::norm(got, host, NORM_INF));
}

TEST(Triton_Resize, RejectsWithoutFallback)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    cuda::GpuMat u8(32, 32, CV_8UC3), dst;
    const auto code = [&](const std::function<void()>& f) -> int {
        try { f(); }
        catch (const cv::Exception& e) { return e.code; }
        return 0;
    };
    // Routes P1 does not serve yet: nothing is submitted, there is no silent substitute.
    const td::RuntimeStats a = td::runtimeStats();
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::resize(u8, dst, Size(50, 50), 0, 0, INTER_CUBIC); }));
    cuda::GpuMat fractional;
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::resize(u8, fractional, Size(20, 21), 0, 0, INTER_AREA); }));
    const td::RuntimeStats b = td::runtimeStats();
    EXPECT_EQ(a.launches, b.launches);
    EXPECT_EQ(a.copies, b.copies);
    // Outside the contract.
    cuda::GpuMat out;
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::resize(u8, out, Size(16, 16), 0, 0, INTER_LANCZOS4); }));
    cuda::GpuMat u16(8, 8, CV_16UC1);
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::resize(u16, out, Size(4, 4)); }));
    cuda::GpuMat two(8, 8, CV_8UC2);
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::resize(two, out, Size(4, 4)); }));
    // Bad arguments.
    EXPECT_EQ(Error::StsBadArg, code([&] { cv::triton::resize(u8, out, Size(), 0, 0); }));
    EXPECT_EQ(Error::StsBadArg, code([&] { cv::triton::resize(u8, u8, Size(16, 16)); }));
    cuda::GpuMat parent(64, 64, CV_8UC1);
    parent.setTo(Scalar::all(0));
    cuda::GpuMat s0 = parent(Rect(0, 0, 32, 32)), d0 = parent(Rect(16, 16, 16, 16));
    EXPECT_EQ(Error::StsBadArg, code([&] { cv::triton::resize(s0, d0, Size(16, 16)); }));
}

}}  // namespace
