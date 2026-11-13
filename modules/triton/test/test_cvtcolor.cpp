// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// cv::triton::cvtColor through its real public entry (GpuMat + C++ planner + binaries from the
// runtime JIT) against OpenCV's CPU cv::cvtColor.
//
// The comparison target is this build's own cv::cvtColor. That is a definite reference, not a
// claim that every build computes the same numbers: integer arithmetic makes a pixel's result
// independent of its position in the row (no reduction, no SIMD width), but that is a property
// of THIS implementation, and a build routed through IPP or a custom HAL is a different one.
// ReferenceBuildIsCharacterised below records that configuration by value; it does not claim to
// prove which code executed. The codes whose CPU result is not even self-consistent (HSV2BGR,
// HLS -- float32, SIMD body vs scalar tail) are refused by the module and absent from this file.
//
// Coverage: every served code on random images (MatchesCpu), on hand-picked boundary colours
// (MatchesCpuOnBoundaries), across block-boundary widths and padded/offset layouts
// (MatchesCpuOnLayouts), plus artifact reuse and in-flight resource retention.

#include "test_helpers.hpp"

#include <chrono>
#include <thread>

#include <cuda.h>

namespace opencv_test { namespace {

using namespace triton_test;

static void CUDA_CB hostSleepCvt(void* ms)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(*static_cast<int*>(ms)));
}
static void blockStream(cuda::Stream& s, int* ms)
{
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchHostFunc(cuda::StreamAccessor::getStream(s), hostSleepCvt, ms));
}
struct Restore
{
    ~Restore() { td::setCacheLimits(64, 256); }
};

struct CvtCase
{
    int code;
    const char* name;
    int scn, dcn;
    // Source/destination depth. Defaulted to CV_8U so every existing row is unchanged; the
    // families that also serve 16U (Bayer, gray, YUV/YCrCb and gray -> colour) get their second
    // depth as extra rows in kCases16U below, and the CPU creates dst at src's depth, so one
    // field describes both sides.
    int depth = CV_8U;
    // 4:2:0 decode takes one CV_8UC1 of (3H/2) x W and produces H x W. The fixture has to build a
    // source of that shape and expect a shorter destination, so the row says so instead of the
    // test guessing from the code number.
    bool yuv420 = false;
};

// Every code the module serves, one row each. A code added to the table without a row here is
// caught by CoversEverySupportedCode below.
const CvtCase kCases[] = {
    {COLOR_BGR2BGRA, "BGR2BGRA", 3, 4},
    {COLOR_BGRA2BGR, "BGRA2BGR", 4, 3},
    {COLOR_BGR2RGBA, "BGR2RGBA", 3, 4},
    {COLOR_RGBA2BGR, "RGBA2BGR", 4, 3},
    {COLOR_BGR2RGB, "BGR2RGB", 3, 3},
    {COLOR_BGRA2RGBA, "BGRA2RGBA", 4, 4},
    {COLOR_BGR2GRAY, "BGR2GRAY", 3, 1},
    {COLOR_RGB2GRAY, "RGB2GRAY", 3, 1},
    {COLOR_BGRA2GRAY, "BGRA2GRAY", 4, 1},
    {COLOR_RGBA2GRAY, "RGBA2GRAY", 4, 1},
    {COLOR_BGR2YCrCb, "BGR2YCrCb", 3, 3},
    {COLOR_RGB2YCrCb, "RGB2YCrCb", 3, 3},
    {COLOR_YCrCb2BGR, "YCrCb2BGR", 3, 3},
    {COLOR_YCrCb2RGB, "YCrCb2RGB", 3, 3},
    {COLOR_BGR2HSV, "BGR2HSV", 3, 3},
    {COLOR_RGB2HSV, "RGB2HSV", 3, 3},
    {COLOR_BGR2HSV_FULL, "BGR2HSV_FULL", 3, 3},
    {COLOR_RGB2HSV_FULL, "RGB2HSV_FULL", 3, 3},
    {COLOR_BGR2YUV, "BGR2YUV", 3, 3},
    {COLOR_RGB2YUV, "RGB2YUV", 3, 3},
    {COLOR_YUV2BGR, "YUV2BGR", 3, 3},
    {COLOR_YUV2RGB, "YUV2RGB", 3, 3},
    // 4:2:2 interleaved: CV_8UC2 in, same size out; only width % 2 == 0 is required.
    {COLOR_YUV2BGR_UYVY, "YUV2BGR_UYVY", 2, 3},
    {COLOR_YUV2RGB_UYVY, "YUV2RGB_UYVY", 2, 3},
    {COLOR_YUV2BGR_YUY2, "YUV2BGR_YUY2", 2, 3},
    {COLOR_YUV2RGB_YUY2, "YUV2RGB_YUY2", 2, 3},
    {COLOR_YUV2BGR_YVYU, "YUV2BGR_YVYU", 2, 3},
    {COLOR_YUV2RGB_YVYU, "YUV2RGB_YVYU", 2, 3},
    // YUV2GRAY_UYVY/YUY2 are extractChannel, not a conversion: they accept ODD widths.
    {COLOR_YUV2GRAY_UYVY, "YUV2GRAY_UYVY", 2, 1},
    {COLOR_YUV2GRAY_YUY2, "YUV2GRAY_YUY2", 2, 1},
    // 4:2:0 semi-planar decode: (3H/2) x W in, H x W out.
    {COLOR_YUV2BGR_NV12, "YUV2BGR_NV12", 1, 3, CV_8U, true},
    {COLOR_YUV2RGB_NV12, "YUV2RGB_NV12", 1, 3, CV_8U, true},
    {COLOR_YUV2BGRA_NV12, "YUV2BGRA_NV12", 1, 4, CV_8U, true},
    {COLOR_YUV2RGBA_NV12, "YUV2RGBA_NV12", 1, 4, CV_8U, true},
    {COLOR_YUV2BGR_NV21, "YUV2BGR_NV21", 1, 3, CV_8U, true},
    {COLOR_YUV2RGB_NV21, "YUV2RGB_NV21", 1, 3, CV_8U, true},
    {COLOR_YUV2BGRA_NV21, "YUV2BGRA_NV21", 1, 4, CV_8U, true},
    {COLOR_YUV2RGBA_NV21, "YUV2RGBA_NV21", 1, 4, CV_8U, true},
    // gray -> color: COLOR_GRAY2RGB is an alias of COLOR_GRAY2BGR (both 8) and
    // COLOR_GRAY2RGBA of COLOR_GRAY2BGRA (both 9), so two rows cover the four names.
    {COLOR_GRAY2BGR, "GRAY2BGR", 1, 3},
    {COLOR_GRAY2BGRA, "GRAY2RGBA", 1, 4},
    // packed 5x5. A "2-channel" source or destination here is one 16-bit word per pixel held in
    // two bytes (CV_8UC2) -- that is what OpenCV 5.x means by a packed 565/555 image, and a
    // 16UC1 Mat of the same pixel count is rejected by the CPU. Both formats, both directions,
    // all four channel combinations and the two gray directions: 20 codes, 20 rows.
    {COLOR_BGR2BGR565, "BGR2BGR565", 3, 2},
    {COLOR_RGB2BGR565, "RGB2BGR565", 3, 2},
    {COLOR_BGR5652BGR, "BGR5652BGR", 2, 3},
    {COLOR_BGR5652RGB, "BGR5652RGB", 2, 3},
    {COLOR_BGRA2BGR565, "BGRA2BGR565", 4, 2},
    {COLOR_RGBA2BGR565, "RGBA2BGR565", 4, 2},
    {COLOR_BGR5652BGRA, "BGR5652BGRA", 2, 4},
    {COLOR_BGR5652RGBA, "BGR5652RGBA", 2, 4},
    {COLOR_GRAY2BGR565, "GRAY2BGR565", 1, 2},
    {COLOR_BGR5652GRAY, "BGR5652GRAY", 2, 1},
    {COLOR_BGR2BGR555, "BGR2BGR555", 3, 2},
    {COLOR_RGB2BGR555, "RGB2BGR555", 3, 2},
    {COLOR_BGR5552BGR, "BGR5552BGR", 2, 3},
    {COLOR_BGR5552RGB, "BGR5552RGB", 2, 3},
    {COLOR_BGRA2BGR555, "BGRA2BGR555", 4, 2},
    {COLOR_RGBA2BGR555, "RGBA2BGR555", 4, 2},
    {COLOR_BGR5552BGRA, "BGR5552BGRA", 2, 4},
    {COLOR_BGR5552RGBA, "BGR5552RGBA", 2, 4},
    {COLOR_GRAY2BGR555, "GRAY2BGR555", 1, 2},
    {COLOR_BGR5552GRAY, "BGR5552GRAY", 2, 1},
    // Bayer bilinear demosaic. One 8U row per served code; the 16U rows live in kCases16U
    // below, because the harness builds its Mat at this row's depth and the depth walk is a
    // separate exercise. Both depths are required by the table, so CoversEverySupportedCode
    // is satisfied by these rows and the 16U test asserts the 16U ones exist.
    {COLOR_BayerBG2BGR, "BayerBG2BGR", 1, 3},
    {COLOR_BayerGB2BGR, "BayerGB2BGR", 1, 3},
    {COLOR_BayerRG2BGR, "BayerRG2BGR", 1, 3},
    {COLOR_BayerGR2BGR, "BayerGR2BGR", 1, 3},
    {COLOR_BayerBG2BGRA, "BayerBG2BGRA", 1, 4},
    {COLOR_BayerBG2GRAY, "BayerBG2GRAY", 1, 1},
    // XYZ and the Lab/Luv families: 18 codes, every one the module serves.
    //
    // The forward sRGB Luv codes are in here even though their CPU path is a trilinear LUT
    // rather than the CIE formula -- reproducing the LUT IS the bit-exact target, and the
    // comparison below is against this build's cv::cvtColor, so a kernel that computed exact
    // CIE Luv would fail here (measured: up to 1 LSB on each of L, u, v).
    //
    // COLOR_LBGR2Luv (76) and COLOR_LRGB2Luv (77) are absent because the module refuses them:
    // the CPU's float forward path answers differently in its SIMD body than in its scalar tail
    // (0.002% of the cube, 1 LSB, switching at width 8), so there is no single result to match.
    // RejectsWithoutFallback asserts they are refused.
    {COLOR_BGR2XYZ, "BGR2XYZ", 3, 3},
    {COLOR_RGB2XYZ, "RGB2XYZ", 3, 3},
    {COLOR_XYZ2BGR, "XYZ2BGR", 3, 3},
    {COLOR_XYZ2RGB, "XYZ2RGB", 3, 3},
    {COLOR_BGR2Lab, "BGR2Lab", 3, 3},
    {COLOR_RGB2Lab, "RGB2Lab", 3, 3},
    {COLOR_Lab2BGR, "Lab2BGR", 3, 3},
    {COLOR_Lab2RGB, "Lab2RGB", 3, 3},
    {COLOR_BGR2Luv, "BGR2Luv", 3, 3},
    {COLOR_RGB2Luv, "RGB2Luv", 3, 3},
    {COLOR_Luv2BGR, "Luv2BGR", 3, 3},
    {COLOR_Luv2RGB, "Luv2RGB", 3, 3},
    {COLOR_LBGR2Lab, "LBGR2Lab", 3, 3},
    {COLOR_LRGB2Lab, "LRGB2Lab", 3, 3},
    {COLOR_Lab2LBGR, "Lab2LBGR", 3, 3},
    {COLOR_Lab2LRGB, "Lab2LRGB", 3, 3},
    {COLOR_Luv2LBGR, "Luv2LBGR", 3, 3},
    {COLOR_Luv2LRGB, "Luv2LRGB", 3, 3},
};

// Every code the module serves at 16U: the Bayer family plus the three integer families whose
// CPU path is the same 15-/14-bit table at both depths (colour <-> gray, YUV, YCrCb). Kept
// separate from kCases so the many tests that iterate kCases keep covering exactly the 8U set
// they were written for; MatchesCpu16U walks this one. One row per served code — a 16U code
// missing here is not caught by CoversEverySupportedCode (that compares the 8U table), so the
// probe in runs/cvtcolor-ext-0001/probe-16u/ sweeps the same list independently.
const CvtCase kCases16U[] = {
    {COLOR_BayerBG2BGR, "BayerBG2BGR", 1, 3, CV_16U},
    {COLOR_BayerGB2BGR, "BayerGB2BGR", 1, 3, CV_16U},
    {COLOR_BayerRG2BGR, "BayerRG2BGR", 1, 3, CV_16U},
    {COLOR_BayerGR2BGR, "BayerGR2BGR", 1, 3, CV_16U},
    {COLOR_BayerBG2BGRA, "BayerBG2BGRA", 1, 4, CV_16U},
    {COLOR_BayerBG2GRAY, "BayerBG2GRAY", 1, 1, CV_16U},
    // The three families whose CPU path is the same integer table at 16U as at 8U.
    {COLOR_BGR2GRAY, "BGR2GRAY", 3, 1, CV_16U},
    {COLOR_RGB2GRAY, "RGB2GRAY", 3, 1, CV_16U},
    {COLOR_BGRA2GRAY, "BGRA2GRAY", 4, 1, CV_16U},
    {COLOR_RGBA2GRAY, "RGBA2GRAY", 4, 1, CV_16U},
    {COLOR_GRAY2BGR, "GRAY2BGR", 1, 3, CV_16U},
    // COLOR_GRAY2BGRA is an alias of COLOR_GRAY2RGBA (both 9), and the table's canonical name
    // for the code is "GRAY2RGBA" -- the route assertion compares against it, so the label here
    // must be that one. The 8U row above already labels it the same way.
    {COLOR_GRAY2BGRA, "GRAY2RGBA", 1, 4, CV_16U},
    {COLOR_BGR2YCrCb, "BGR2YCrCb", 3, 3, CV_16U},
    {COLOR_RGB2YCrCb, "RGB2YCrCb", 3, 3, CV_16U},
    {COLOR_YCrCb2BGR, "YCrCb2BGR", 3, 3, CV_16U},
    {COLOR_YCrCb2RGB, "YCrCb2RGB", 3, 3, CV_16U},
    {COLOR_BGR2YUV, "BGR2YUV", 3, 3, CV_16U},
    {COLOR_RGB2YUV, "RGB2YUV", 3, 3, CV_16U},
    {COLOR_YUV2BGR, "YUV2BGR", 3, 3, CV_16U},
    {COLOR_YUV2RGB, "YUV2RGB", 3, 3, CV_16U},
};

// Every admitted 32F numeric code. Copies compare exactly; arithmetic uses the
// per-pixel input-channel scale bound. Coverage is checked against the planner table.
const CvtCase kCases32F[] = {
    {COLOR_BGR2BGRA, "BGR2BGRA", 3, 4, CV_32F},
    {COLOR_BGRA2BGR, "BGRA2BGR", 4, 3, CV_32F},
    {COLOR_BGR2RGBA, "BGR2RGBA", 3, 4, CV_32F},
    {COLOR_RGBA2BGR, "RGBA2BGR", 4, 3, CV_32F},
    {COLOR_BGR2RGB, "BGR2RGB", 3, 3, CV_32F},
    {COLOR_BGRA2RGBA, "BGRA2RGBA", 4, 4, CV_32F},
    {COLOR_GRAY2BGR, "GRAY2BGR", 1, 3, CV_32F},
    {COLOR_GRAY2BGRA, "GRAY2BGRA", 1, 4, CV_32F},
    // The 32F ARITHMETIC codes. These are NOT bit-exact and are not compared for equality: the CPU
    // has several answers (fused body for W>=16, fused tail for W<16, unfused build), so they are
    // checked against the measured bound |got-ref| <= 8*2^-24*max(1,|ref|,max|input|).
    {COLOR_BGR2GRAY, "BGR2GRAY", 3, 1, CV_32F},
    {COLOR_RGB2GRAY, "RGB2GRAY", 3, 1, CV_32F},
    {COLOR_BGRA2GRAY, "BGRA2GRAY", 4, 1, CV_32F},
    {COLOR_RGBA2GRAY, "RGBA2GRAY", 4, 1, CV_32F},
    {COLOR_BGR2YCrCb, "BGR2YCrCb", 3, 3, CV_32F},
    {COLOR_RGB2YCrCb, "RGB2YCrCb", 3, 3, CV_32F},
    {COLOR_YCrCb2BGR, "YCrCb2BGR", 3, 3, CV_32F},
    {COLOR_YCrCb2RGB, "YCrCb2RGB", 3, 3, CV_32F},
    {COLOR_BGR2YUV, "BGR2YUV", 3, 3, CV_32F},
    {COLOR_RGB2YUV, "RGB2YUV", 3, 3, CV_32F},
    {COLOR_YUV2BGR, "YUV2BGR", 3, 3, CV_32F},
    {COLOR_YUV2RGB, "YUV2RGB", 3, 3, CV_32F},
};

static std::string label(const CvtCase& c, Size sz)
{
    return cv::format("%s %dx%d", c.name, sz.width, sz.height);
}

// Sizes chosen to hit the interesting boundaries of a flat pass over pixels: smaller than one
// BLOCK, exactly one BLOCK, one more, a row that is not a multiple of the block, and a real
// image size. The kernel's masking is the thing under test at the small end.
const Size kSizes[] = {Size(1, 1), Size(13, 7), Size(256, 1), Size(257, 3), Size(97, 61), Size(640, 480)};

// The 4:2:0 codes have their own size rule (even width, source height divisible by 3), so the
// general list above contains shapes the CPU itself rejects. These are the same boundary ideas
// (smaller than a block, exactly a block, not a multiple of one, a real size) restricted to legal
// 4:2:0 sources.
const Size kSizes420[] = {Size(2, 2), Size(4, 2), Size(16, 4), Size(130, 10), Size(256, 6), Size(640, 480)};

TEST(Triton_CvtColor, MatchesCpu)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    uint64 seed = 1;
    cuda::Stream stream;
    for (const CvtCase& c : kCases)
        for (const Size& sz : kSizes)
        {
            if (c.yuv420) continue;                  // covered by MatchesCpu420 with its own sizes
            // The 4:2:2 CONVERSIONS require an even width; the CPU asserts
            // (`sz.width % 2 == 0`, color.simd_helpers.hpp:119) rather than producing an answer,
            // so an odd width from the shared list is not a legal input for them. The two
            // YUV2GRAY_* codes are the exception -- extractChannel has no parity assert -- so the
            // test keeps them on the full list.
            const bool yuv422conv = c.scn == 2 && c.dcn != 1;
            if (yuv422conv && (sz.width % 2) != 0) continue;
            SCOPED_TRACE(label(c, sz));
            // A 4:2:0 source is (3H/2) x W and decodes to H x W, so the fixture builds the taller
            // buffer and expects the shorter result. Everything else keeps the size, and the
            // assertion below still pins that.
            const Size srcSz = sz;
            const Mat host = randomImage(srcSz, CV_MAKETYPE(c.depth, c.scn), seed++);
            cuda::GpuMat src(host), dst;
            cv::triton::cvtColor(src, dst, c.code, stream);
            stream.waitForCompletion();
            EXPECT_STREQ(c.name, td::runtimeStats().last_route.c_str());
            Mat got;
            dst.download(got);
            ASSERT_EQ(got.size(), host.size());
            ASSERT_EQ(got.channels(), c.dcn);
            // Integer arithmetic on both sides: exact, not "within a tolerance".
            EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, c.code), NORM_INF));
        }
}

// Full finite 32F coverage, including mixed per-pixel scales.
TEST(Triton_CvtColor, MatchesCpu32F)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    uint64 seed = 7000;
    cuda::Stream stream;
    for (const CvtCase& c : kCases32F)
        for (const Size& sz : kSizes)
        {
            SCOPED_TRACE(cv::format("%s %dx%d", c.name, sz.width, sz.height));
            Mat host = randomImage(sz, CV_MAKETYPE(CV_32F, c.scn), seed++);
            // Adjacent small and large pixels expose a mistakenly image-global bound.
            for (int y = 0; y < host.rows; ++y)
                for (int x = 0; x < host.cols; ++x)
                    if ((x + y) % 3 == 0)
                        for (int k = 0; k < c.scn; ++k)
                            host.ptr<float>(y)[x*c.scn+k] *= 32767.5f;
            cuda::GpuMat src(host), dst;
            cv::triton::cvtColor(src, dst, c.code, stream);
            stream.waitForCompletion();
            // Not compared against c.name: several codes are enum ALIASES of one another
            // (GRAY2BGRA == GRAY2RGBA == 9, BGR2RGB/BGR2BGRA have RGB-order twins), so the route
            // reports the name of whichever row the table holds, not the caller's spelling.
            // MatchesCpu's name check does hold for the non-aliased codes it covers.
            Mat got;
            dst.download(got);
            ASSERT_EQ(got.size(), host.size());
            ASSERT_EQ(got.type(), CV_MAKETYPE(CV_32F, c.dcn));
            // Two different claims, and the test must not blur them. The bit-copy codes write each
            // source float into a destination slot unchanged, so equality holds (including the
            // alpha = 1.0f the inserted channel gets at 32F -- it is 255 at 8U, which is why the
            // kernel derives it). The arithmetic codes have several CPU answers, so they are
            // checked against the measured bound, not for equality.
            const bool arithmetic = c.code == COLOR_BGR2GRAY || c.code == COLOR_RGB2GRAY ||
                                    c.code == COLOR_BGRA2GRAY || c.code == COLOR_RGBA2GRAY ||
                                    c.code == COLOR_BGR2YCrCb || c.code == COLOR_RGB2YCrCb ||
                                    c.code == COLOR_YCrCb2BGR || c.code == COLOR_YCrCb2RGB ||
                                    c.code == COLOR_BGR2YUV || c.code == COLOR_RGB2YUV ||
                                    c.code == COLOR_YUV2BGR || c.code == COLOR_YUV2RGB;
            const Mat ref = cpuCvt(host, c.code);
            if (!arithmetic)
            {
                EXPECT_EQ(0.0, cvtest::norm(got, ref, NORM_INF));
            }
            else
            {
                // |got-ref| <= 8 * 2^-24 * max(1, |ref|, max|input|). max|input| is required: a
                // chroma channel can be computed from 6.5e4-scale operands and land near zero, so
                // a |ref|-only normaliser over-claims by orders of magnitude (contract-32f.md 5.2).
                const double k = 8.0 * std::ldexp(1.0, -24);
                double worst = 0;
                for (int y = 0; y < got.rows; ++y)
                    for (int x = 0; x < got.cols * got.channels(); ++x)
                    {
                        const double r = ref.ptr<float>(y)[x], g = got.ptr<float>(y)[x];
                        double maxin = 0;
                        const int pixel = x / got.channels();
                        for (int ch = 0; ch < host.channels(); ++ch)
                            maxin = std::max(maxin, std::fabs(double(host.ptr<float>(y)[pixel*host.channels()+ch])));
                        ASSERT_TRUE(std::isfinite(r) && std::isfinite(g));
                        const double allow = k * std::max(std::max(1.0, std::fabs(r)), maxin);
                        worst = std::max(worst, std::fabs(g - r) / allow);
                    }
                EXPECT_LE(worst, 1.0) << "outside the measured 32F bound (k=" << worst << ")";
            }
        }
}

// The 4:2:0 decode half. A separate test rather than more rows in MatchesCpu because the size
// rule differs: the source is (3H/2) x W and the result is H x W, and the CPU itself rejects
// shapes from the general list (it asserts an even width and a height divisible by 3).
TEST(Triton_CvtColor, MatchesCpu420)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    uint64 seed = 900;
    cuda::Stream stream;
    for (const CvtCase& c : kCases)
    {
        if (!c.yuv420) continue;
        for (const Size& sz : kSizes420)
        {
            SCOPED_TRACE(cv::format("%s %dx%d", c.name, sz.width, sz.height));
            const Mat host = randomImage(Size(sz.width, sz.height * 3 / 2),
                                         CV_MAKETYPE(c.depth, c.scn), seed++);
            cuda::GpuMat src(host), dst;
            cv::triton::cvtColor(src, dst, c.code, stream);
            stream.waitForCompletion();
            EXPECT_STREQ(c.name, td::runtimeStats().last_route.c_str());
            Mat got;
            dst.download(got);
            // The size really did change: this is the property that distinguishes the family, and
            // asserting it here is what would catch a regression to "dst created at src.size()".
            ASSERT_EQ(got.size(), sz) << "4:2:0 decode must produce H x W, not the source shape";
            ASSERT_EQ(got.channels(), c.dcn);
            EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, c.code), NORM_INF));
        }
    }
}

// The 16U half. Bayer is the only family the module serves at a second depth, and it is the same
// integer algorithm there (demosaicing.cpp:1796-1803), so "exact" is the same claim and the same
// comparison; only the Mat type changes. Sizes are reused so an addressing slip shows up at the
// same places it would at 8U.
TEST(Triton_CvtColor, MatchesCpu16U)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    uint64 seed = 4000;
    cuda::Stream stream;
    for (const CvtCase& c : kCases16U)
        for (const Size& sz : kSizes)
        {
            SCOPED_TRACE(label(c, sz) + " (16U)");
            const Mat host = randomImage(sz, CV_MAKETYPE(c.depth, c.scn), seed++);
            // Prove the fixture is really 16U with the full range filled: a CV_8U image here
            // would make every comparison an 8U comparison and pass while covering nothing.
            ASSERT_EQ(CV_16U, host.depth());
            double lo, hi;
            cv::minMaxLoc(host, &lo, &hi);
            ASSERT_GT(hi, 256.0) << "fixture: 16U values never leave the 8-bit range";
            cuda::GpuMat src(host), dst;
            cv::triton::cvtColor(src, dst, c.code, stream);
            stream.waitForCompletion();
            EXPECT_STREQ(c.name, td::runtimeStats().last_route.c_str());
            Mat got;
            dst.download(got);
            ASSERT_EQ(got.size(), host.size());
            ASSERT_EQ(got.channels(), c.dcn);
            EXPECT_EQ(CV_16U, got.depth()) << "the destination changed depth";
            EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, c.code), NORM_INF));
        }
}

// The default stream and output reuse: a second call into the same destination must not
// reallocate it, and must give the same answer.
TEST(Triton_CvtColor, ReusesOutputOnDefaultStream)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    const Mat host = randomImage(Size(131, 97), CV_8UC3, 7);
    cuda::GpuMat src(host), dst;
    cv::triton::cvtColor(src, dst, COLOR_BGR2YUV);
    cuda::Stream::Null().waitForCompletion();
    const void* before = dst.cudaPtr();
    cv::triton::cvtColor(src, dst, COLOR_BGR2YUV);
    cuda::Stream::Null().waitForCompletion();
    EXPECT_EQ(before, dst.cudaPtr()) << "the output was reallocated on the second call";
    Mat got;
    dst.download(got);
    EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, COLOR_BGR2YUV), NORM_INF));
}

// ROI source and destination: the result equals the CPU conversion of the source ROI, and
// nothing outside the destination ROI is written.
TEST(Triton_CvtColor, RoutesRoi)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    uint64 seed = 5000;
    cuda::Stream stream;
    for (const CvtCase& c : kCases)
    {
        SCOPED_TRACE(c.name);
        // 67x43 was fine while every served code accepted odd widths. Two of the CPU's own source
        // rules then arrived: a 4:2:2 CONVERSION needs an even width (color.simd_helpers.hpp:119)
        // and a 4:2:0 source needs an even width with a height divisible by 3. 72x45 is the old
        // shape's neighbourhood and satisfies both, so one fixture still covers every code and
        // the ROI/pitch path this test exists for is unchanged.
        // A 4:2:0 SOURCE is 3/2 the height of its output, so the two views are not the same
        // shape for those codes (the module derives dst's size from the code, exactly as the
        // CPU's CvtHelper does). Every other code keeps src and dst the same size.
        const Size sz(72, 45);
        const Size dsz = (c.yuv420) ? Size(sz.width, sz.height * 2 / 3) : sz;
        const Mat host = randomImage(sz, CV_MAKETYPE(c.depth, c.scn), seed++);
        cuda::GpuMat srcParent, dstParent;
        cuda::GpuMat src = roiOf(srcParent, sz, CV_MAKETYPE(c.depth, c.scn), Point(3, 1), 7, 3, 0.0);
        cuda::GpuMat dst = roiOf(dstParent, dsz, CV_MAKETYPE(c.depth, c.dcn), Point(5, 2), 9, 5, 77.0);
        src.upload(host);
        const void* before = dst.cudaPtr();
        cv::triton::cvtColor(src, dst, c.code, stream);
        stream.waitForCompletion();
        EXPECT_EQ(before, dst.cudaPtr()) << "the output was reallocated";
        Mat got;
        dst.download(got);
        EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, c.code), NORM_INF));
        EXPECT_TRUE(guardsIntact(dstParent, Rect(Point(5, 2), dsz), 77.0)) << "wrote outside the dst ROI";
    }
}

// Value boundaries, not random images. Random 8-bit data almost never lands on the inputs where
// a fixed-point formula saturates, where a hue branch is exactly zero, or where two channels tie
// for the maximum -- which is where the CPU's own code takes its special cases. Each case below
// is a set of literal colours chosen for that, so a failure names one real input.
struct BoundaryCase { const char* name; std::vector<Vec3b> colours; };

static std::vector<BoundaryCase> boundaryCases()
{
    std::vector<BoundaryCase> v;
    v.push_back({"extremes", {Vec3b(0,0,0), Vec3b(255,255,255), Vec3b(0,0,255), Vec3b(255,0,0),
                              Vec3b(0,255,0), Vec3b(255,255,0), Vec3b(0,255,255), Vec3b(255,0,255)}});
    v.push_back({"grey", {Vec3b(1,1,1), Vec3b(2,2,2), Vec3b(127,127,127), Vec3b(128,128,128),
                          Vec3b(254,254,254), Vec3b(255,255,255)}});
    // Channel maxima equal: d == 0, so the CPU divides by a zero diff and the s/h branches are
    // the d==0 special case, not a formula.
    v.push_back({"tie_max", {Vec3b(10,10,10), Vec3b(200,200,200), Vec3b(0,0,0)}});
    // Hue at each primary/secondary sector and each boundary between sectors.
    v.push_back({"hue_edges", {Vec3b(255,0,0), Vec3b(0,255,0), Vec3b(0,0,255), Vec3b(255,255,0),
                               Vec3b(0,255,255), Vec3b(255,0,255), Vec3b(1,0,0), Vec3b(0,1,0),
                               Vec3b(0,0,1), Vec3b(255,1,0), Vec3b(254,255,0), Vec3b(0,255,254)}});
    // Low saturation: d == 1 and d == 2 at both ends of V -- the case where s and h are most
    // sensitive to the table entries and where saturation_cast is most likely to clip.
    v.push_back({"low_sat", {Vec3b(100,100,101), Vec3b(100,101,100), Vec3b(101,100,100),
                             Vec3b(254,255,255), Vec3b(0,0,1), Vec3b(1,1,0)}});
    // Saturated chroma with a large luma spread: the YUV/YCrCb chroma terms reach their clamp.
    v.push_back({"chroma_clamp", {Vec3b(0,0,255), Vec3b(255,0,0), Vec3b(0,255,0),
                                  Vec3b(0,128,255), Vec3b(255,128,0)}});
    // Grey and near-grey YUV: the inverse transform's chroma is 128 +- 0 or 1, where the
    // (v - 128) rounding of the inverse path is decided.
    return v;
}

TEST(Triton_CvtColor, MatchesCpuOnBoundaries)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    cuda::Stream stream;
    const Size sz(61, 37);
    for (const CvtCase& c : kCases)
    {
        // Codes whose source shape is constrained (odd width forbidden, or a 4:2:0 height rule)
        // cannot run on this fixture's 61x37 buffer: the CPU itself asserts. They have their own
        // dedicated coverage (MatchesCpu420, and the 4:2:2 rows in MatchesCpu).
        if (c.yuv420 || c.scn == 2) continue;
        for (const BoundaryCase& b : boundaryCases())
        {
            SCOPED_TRACE(label(c, sz) + " / " + b.name);
            Mat host(sz, CV_MAKETYPE(c.depth, c.scn), Scalar::all(0));
            // Written through a byte pointer at the code's channel count. An earlier version
            // wrote `Vec3b px; px[3] = ...` for the 4-channel codes: that is three bytes past the
            // end of px, and `at<Vec3b>` on a 4-channel Mat strides rows by 3*cols instead of
            // 4*cols. Both are undefined behaviour that happened not to crash, so the boundary
            // test could pass without ever placing the intended pixel.
            for (int i = 0; i < (int)b.colours.size(); ++i)
            {
                const Vec3b& v = b.colours[i];
                uchar* px = host.ptr(i / sz.width) + (i % sz.width) * c.scn;
                // Only `scn` bytes belong to this pixel. Writing px[1]/px[2] on a 1-channel image
                // spills into the next pixel, which is the same class of out-of-bounds the comment
                // above describes -- and the fixture self-check below is what caught it.
                px[0] = v[0];
                if (c.scn > 1) px[1] = v[1];
                if (c.scn > 2) px[2] = v[2];
                if (c.scn == 4) px[3] = (uchar)(i * 37);           // alpha: carried, never computed
            }
            // Prove the fixture before using it: the colours really are where they were meant
            // to go, and (for the 4-channel codes) alpha really is set. Without this the test
            // compares the kernel with the CPU on the same wrong input and passes.
            for (int i = 0; i < (int)b.colours.size(); ++i)
            {
                const Vec3b& v = b.colours[i];
                const uchar* px = host.ptr(i / sz.width) + (i % sz.width) * c.scn;
                ASSERT_EQ(v[0], px[0]) << "fixture: colour " << i << " byte 0";
                if (c.scn > 1) ASSERT_EQ(v[1], px[1]) << "fixture: colour " << i << " byte 1";
                if (c.scn > 2) ASSERT_EQ(v[2], px[2]) << "fixture: colour " << i << " byte 2";
                if (c.scn == 4) ASSERT_EQ((uchar)(i * 37), px[3]) << "fixture: alpha " << i;
            }
            // The inverse codes read [Y, Cr, Cb] / [Y, U, V]: sweeping their input space is the
            // same exercise as sweeping a colour, so feed the same literal tuples either way.
            cuda::GpuMat src(host), dst;
            cv::triton::cvtColor(src, dst, c.code, stream);
            stream.waitForCompletion();
            Mat got;
            dst.download(got);
            EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, c.code), NORM_INF));
        }
    }
}

// Every size near a block boundary and every source/destination geometry whose row pitch is not
// the natural one. The kernel addresses row*pitch + col*ch, so the pitch is the thing that can be
// wrong, and it is wrong exactly at the sizes where the allocator pads.
TEST(Triton_CvtColor, MatchesCpuOnLayouts)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    const int block = 256;                      // kf_cvtcolor::block_size: at cols-1, cols, cols+1
    const Size sizes[] = {Size(block - 1, 5), Size(block, 5), Size(block + 1, 5),
                          Size(1, 257), Size(3, 1), Size(3, 2), Size(4, 3)};
    uint64 seed = 900;
    for (const CvtCase& c : kCases)
        for (const Size& sz : sizes)
            for (int padX : {0, 1, 5, 17})
            {
                // The CPU REJECTS these shapes itself, so there is no reference answer to compare
                // against and the failure would be in the harness, not the kernel: 4:2:0 needs an
                // even width and a source height divisible by 3, and a 4:2:2 CONVERSION needs an
                // even width (color.simd_helpers.hpp:119). The YUV2GRAY_* 4:2:2 codes are
                // extractChannel and stay in.
                if (c.yuv420) continue;
                if (c.scn == 2 && c.dcn != 1 && (sz.width % 2) != 0) continue;
                SCOPED_TRACE(label(c, sz) + cv::format(" padX=%d", padX));
                const Mat host = randomImage(sz, CV_MAKETYPE(c.depth, c.scn), seed++);
                cuda::GpuMat srcParent, dstParent;
                cuda::GpuMat src = roiOf(srcParent, sz, CV_MAKETYPE(c.depth, c.scn),
                                         Point(0, 0), sz.width + padX, 2, 0.0);
                src.upload(host);
                // The destination is an offset ROI inside a padded parent, so both its pitch and
                // its begin pointer differ from src's, and the guard outside it must survive.
                cuda::GpuMat dst = roiOf(dstParent, sz, CV_MAKETYPE(c.depth, c.dcn),
                                         Point(3, 1), sz.width + padX + 9, 4, 77.0);
                cv::triton::cvtColor(src, dst, c.code);
                cuda::Stream::Null().waitForCompletion();
                Mat got;
                dst.download(got);
                EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, c.code), NORM_INF));
                EXPECT_TRUE(guardsIntact(dstParent, Rect(Point(3, 1), sz), 77.0))
                    << "wrote outside the destination ROI";
            }
}

// Everything the module must refuse, with the error code it must use. A call that returns has
// run the Triton kernel: there is no quiet fallback to the CPU implementation.
TEST(Triton_CvtColor, RejectsWithoutFallback)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    cuda::GpuMat src(32, 32, CV_8UC3), dst;
    const auto code = [&](const std::function<void()>& f) -> int {
        const auto before = td::runtimeStats();
        try { f(); }
        catch (const cv::Exception& e) {
            EXPECT_EQ(before.launches, td::runtimeStats().launches)
                << "a rejected request launched work";
            return e.code;
        }
        return 0;
    };
    // The float32 conversions: not served, refused by name rather than under a tolerance.
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(src, dst, COLOR_HSV2BGR); }));
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(src, dst, COLOR_BGR2HLS); }));
    // The linear-Luv FORWARD codes. Not float32-in-general, and not "hard": the CPU's own answer
    // depends on the image width (SIMD body vs scalar tail, 0.002% of the 2^24 cube, 1 LSB,
    // switching at width 8), so the module has nothing single-valued to reproduce. Asserted here
    // so removing the refusal has to delete an explicit statement rather than slip through.
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(src, dst, COLOR_LBGR2Luv); }));
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(src, dst, COLOR_LRGB2Luv); }));
    // 32F is served only where the module has a stated contract: the bit-copy codes and the
    // three arithmetic families (colour->gray, YCrCb, YUV). Everything else at 32F is refused.
    cuda::GpuMat f32(32, 32, CV_32FC3);
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(f32, dst, COLOR_BGR2XYZ); }));
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(f32, dst, COLOR_BGR2HSV); }));
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(f32, dst, COLOR_BGR2Lab); }));
    // ...and the arithmetic ones really are served, not merely not-refused.
    {
        cuda::GpuMat g32, d32;
        EXPECT_NO_THROW(cv::triton::cvtColor(f32, d32, COLOR_BGR2GRAY));
        EXPECT_EQ(CV_32FC1, d32.type());
    }
    // extractChannel itself supports floats, but cvtColorYUV2Gray_ch explicitly
    // restricts these conversion codes to CV_8U. Both real public entries reject.
    Mat extractHost(3, 7, CV_32FC2, Scalar::all(1));
    cuda::GpuMat extractGpu(extractHost);
    for (int extractCode : {COLOR_YUV2GRAY_UYVY, COLOR_YUV2GRAY_YUY2})
    {
        Mat ref;
        EXPECT_THROW(cv::cvtColor(extractHost, ref, extractCode), cv::Exception);
        EXPECT_EQ(Error::StsUnsupportedFormat,
                  code([&] { cv::triton::cvtColor(extractGpu, dst, extractCode); }));
    }
    // A source with the wrong channel count for the code.
    cuda::GpuMat c1(32, 32, CV_8UC1);
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(c1, dst, COLOR_BGR2GRAY); }));
    // In-place.
    cuda::GpuMat inplace(32, 32, CV_8UC3);
    EXPECT_EQ(Error::StsBadArg, code([&] { cv::triton::cvtColor(inplace, inplace, COLOR_BGR2RGB); }));
    // Bayer VNG: 8U-only on the CPU AND a gradient search rather than the bilinear taps, so it
    // is not in the table. Refused, not served as "close enough" (demosaicing.cpp:1806 rejects
    // 16U for it and :1825-1830 routes it to a different kernel entirely).
    cuda::GpuMat bayer8(32, 32, CV_8UC1);
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(bayer8, dst, COLOR_BayerBG2BGR_VNG); }));
    // The Bayer family at a depth the CPU does not support there either: 32F.
    cuda::GpuMat bayer32f(32, 32, CV_32FC1);
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(bayer32f, dst, COLOR_BayerBG2BGR); }));
    // 16U is admitted for Bayer, gray, YUV/YCrCb and gray -> colour -- the families whose CPU
    // path is the same integer formula at both depths. Everything else is still 8U-only, and
    // these are the cases that would slip through if the gate were widened to "any op at 16U":
    // a colour permute (excluded by the scn == 1 condition), HSV, and the packed format.
    cuda::GpuMat g16(32, 32, CV_16UC3);
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(g16, dst, COLOR_BGR2BGRA); }));
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(g16, dst, COLOR_BGR2HSV); }));
    cuda::GpuMat g16packed(32, 32, CV_16UC2);
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(g16packed, dst, COLOR_BGR5652BGR); }));
    // ... and a served code really is served at 16U through the same entry, so the assertion
    // above cannot pass by everything being refused.
    cv::triton::cvtColor(g16, dst, COLOR_BGR2GRAY);
    cuda::Stream::Null().waitForCompletion();
    EXPECT_EQ(CV_16U, dst.depth()) << "BGR2GRAY at 16U did not produce a 16U destination";
    // ... and src/dst depths must agree: the CPU creates dst at src's depth, so a mixed pair is
    // not something it can produce.
    // src/dst depth cannot disagree THROUGH THIS ENTRY: the adapter re-creates dst as
    // CV_MAKETYPE(src.depth(), dcn), so an existing 8U dst handed a 16U source is replaced, not
    // mixed. normalize()'s matching-depth check is therefore about the planner contract, not
    // something a caller can trip here -- and the observable behaviour is the depth following
    // the SOURCE.
    cuda::GpuMat dst8(32, 32, CV_8UC3);
    cuda::GpuMat bayer16(32, 32, CV_16UC1);
    cv::triton::cvtColor(bayer16, dst8, COLOR_BayerBG2BGR);
    cuda::Stream::Null().waitForCompletion();
    EXPECT_EQ(CV_16U, dst8.depth()) << "the destination did not follow the source depth";
    // A Bayer source must be 1-channel: 8UC3 is rejected by the CPU too
    // (CV_Assert(scn == 1 && (dcn == 3 || dcn == 4), demosaicing.cpp:1787).
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(src, dst, COLOR_BayerBG2BGR); }));
    // A destination that already exists with the wrong channel count is made to fit, not
    // rejected: BGR2BGRA into a 3-channel buffer must produce 4 channels.
    cuda::GpuMat wrong(32, 32, CV_8UC3);
    cv::triton::cvtColor(src, wrong, COLOR_BGR2BGRA);
    cuda::Stream::Null().waitForCompletion();
    EXPECT_EQ(4, wrong.channels());
}

// Malformed shapes must stop before compilation or launch. Do not exercise an
// unsafe odd-width 422 kernel to discover the missing validation.
TEST(Triton_CvtColor, RejectsMalformedVideoBeforeLaunch)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    const int decode422[] = {COLOR_YUV2BGR_UYVY, COLOR_YUV2RGB_UYVY,
                            COLOR_YUV2BGR_YUY2, COLOR_YUV2RGB_YUY2,
                            COLOR_YUV2BGR_YVYU, COLOR_YUV2RGB_YVYU};
    for (int code : decode422)
        for (int width : {1, 3, 31})
        {
            SCOPED_TRACE(cv::format("code=%d width=%d", code, width));
            cuda::GpuMat src(1, width, CV_8UC2), dst;
            const auto before = td::runtimeStats();
            EXPECT_THROW(cv::triton::cvtColor(src, dst, code), cv::Exception);
            EXPECT_EQ(before.launches, td::runtimeStats().launches);
        }
    for (const Size& size : {Size(31, 24), Size(32, 23)})
    {
        cuda::GpuMat src(size, CV_8UC1), dst;
        const auto before = td::runtimeStats();
        EXPECT_THROW(cv::triton::cvtColor(src, dst, COLOR_YUV2BGR_NV12), cv::Exception);
        EXPECT_EQ(before.launches, td::runtimeStats().launches);
    }
    // Odd-width Y extraction remains legal, and really launches.
    Mat host = randomImage(Size(3, 1), CV_8UC2, 12);
    cuda::GpuMat src(host), dst;
    const auto before = td::runtimeStats();
    cv::triton::cvtColor(src, dst, COLOR_YUV2GRAY_UYVY);
    cuda::Stream::Null().waitForCompletion();
    EXPECT_EQ(before.launches + 1, td::runtimeStats().launches);
    Mat got; dst.download(got);
    EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, COLOR_YUV2GRAY_UYVY), NORM_INF));
}

// Which build is being compared against, and what that does and does not establish.
//
// An earlier version of this test claimed "the reference is the generic CPU path" on the
// strength of color.cpp having no CALL_HAL. That was wrong: the top-level entries call
// hal::cvtBGRtoBGR / cvtBGRtoYUV / cvtBGRtoHSV in color_rgb.dispatch.cpp and friends, and
// those do have CALL_HAL. What this test can honestly do is check the configuration by VALUE
// -- a custom HAL would have to be compiled in and would say so -- and refuse to draw a
// numeric conclusion on a build it cannot characterise. It cannot observe a runtime-registered
// HAL, so it does not claim to.
static std::string buildField(const std::string& key)
{
    const std::string info = cv::getBuildInformation();
    for (std::size_t p = 0; p < info.size();)
    {
        const std::size_t e = info.find('\n', p);
        const std::string line = info.substr(p, e == std::string::npos ? std::string::npos : e - p);
        const std::size_t c = line.find(':');
        if (c != std::string::npos)
        {
            std::string k = line.substr(0, c);
            const std::size_t b = k.find_first_not_of(" \t");
            std::string v = line.substr(c + 1);
            const std::size_t vb = v.find_first_not_of(" \t");
            if (b != std::string::npos && k.substr(b) == key)
                return vb == std::string::npos ? std::string() : v.substr(vb);
        }
        if (e == std::string::npos) break;
        p = e + 1;
    }
    return std::string();
}

TEST(Triton_CvtColor, ReferenceBuildIsCharacterised)
{
    // Recorded, not inferred: these are the fields the claim depends on, read by value. A build
    // that disagrees is reported with the value it had, so a failure says what changed.
    const std::string ipp = buildField("Intel IPP");
    const std::string hal = buildField("Custom HAL");
    const std::string base = buildField("Baseline");
    if (base.empty())
        throw SkipTestException("cannot read the build configuration");

    // This test RECORDS the configuration; it must not fail a build for being configured
    // differently. A contrib module is built in whatever configuration the packager chose, and
    // IPP or a vendor HAL are legitimate there. What those configurations invalidate is the
    // *claim* that this build's cv::cvtColor is the generic path the module was verified
    // against -- which is exactly what the numeric tests need to know, so the finding is
    // published as a skipped diagnostic rather than an assertion failure.
    //
    // cvtColor numeric tests compare against this build's own cv::cvtColor, not
    // the resize portable oracle. A different reference build needs revalidation;
    // this diagnostic cannot silently confer the previous build's numeric scope.
    const bool isReference = (ipp == "NO" || ipp.empty()) && hal.compare(0, 2, "NO") == 0
                             && !cv::ipp::useIPP();
    std::cout << "cvtColor reference build: Baseline=" << base << " Intel IPP=" << ipp
              << " Custom HAL=" << hal << " ipp_active=" << (cv::ipp::useIPP() ? 1 : 0)
              << " is_reference=" << (isReference ? 1 : 0) << std::endl;
    if (!isReference)
        throw SkipTestException(
            "this build is not the generic reference configuration (Intel IPP: " + ipp +
            ", Custom HAL: " + hal + "); the module's numeric evidence was gathered on a build "
            "without them, so a host cv::cvtColor comparison here is of unknown scope. The "
            "numeric comparisons use this build and do not inherit earlier reference qualification");
}

// The same kernel reused: no second compile, no second module load, the tables uploaded once.
// The counters start at zero but the caches do not -- an earlier test in this binary may have
// already uploaded these tables -- so the invariant is "six calls upload at most once and hit
// otherwise", not "the first call uploads".
TEST(Triton_CvtColor, ReusesCompiledArtifacts)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    td::reapCompleted();
    td::resetRuntimeStats();
    cuda::Stream stream;
    const Mat host = randomImage(Size(203, 111), CV_8UC3, 77);
    cuda::GpuMat src(host), dst;
    for (int i = 0; i < 6; ++i)
    {
        cv::triton::cvtColor(src, dst, COLOR_BGR2YUV, stream);
        stream.waitForCompletion();
    }
    const td::RuntimeStats st = td::runtimeStats();
    EXPECT_EQ(6, st.launches);
    EXPECT_LE(st.module_loads, 1) << "reloaded the module on every call";
    EXPECT_LE(st.helper_compiles, 1) << "compiled again";
    EXPECT_GE(st.module_hits, 5) << "the module cache was not used";
    EXPECT_LE(st.table_uploads, 1) << "re-uploaded the div tables";
    EXPECT_GE(st.table_hits, 5) << "the table cache was not used";
    EXPECT_EQ(0, st.disk_rejected);
    Mat got;
    dst.download(got);
    EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, COLOR_BGR2YUV), NORM_INF));
}

// Asynchronous keep-alive, with a deterministic eviction of the resources the pending launch
// is holding. Three earlier attempts are worth recording, because each passed or failed for the
// wrong reason:
//
//   * `frees_after >= frees_before` -- satisfied by any release of anything. Cannot fail.
//   * shrinking the cache to 1/1 and asserting global frees == 0 -- but the shrink also frees
//     unrelated entries, which is legitimate, so the assertion was wrong, not the runtime.
//   * shrinking to 1/1 and then calling -- the shrink had already evicted the module the call
//     needed, so the call reloaded it; `cuModuleLoad` drains the context, the "pending" launch
//     was synchronised away, and the measured call blocked for the whole 400 ms block.
//
// The construction below removes all three: the cache is first emptied of unrelated entries,
// exactly two codes are warmed, and then the two evictions are produced deliberately -- a table
// eviction by a second code, a module eviction by a limit shrink -- with the counters checked
// in between. Statistics are reset after the setup evictions, so a free counted later can only
// belong to the resources under test.
TEST(Triton_CvtColor, HoldsResourcesWhileInFlight)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    Restore restore;
    const Mat host = randomImage(Size(157, 93), CV_8UC3, 91);
    const Mat want = cpuCvt(host, COLOR_BGR2HSV);
    cuda::GpuMat src(host), dst, other;
    cuda::Stream a(cudaStreamNonBlocking), b(cudaStreamNonBlocking);

    // Empty the caches of whatever earlier tests left, then let them grow again.
    td::setCacheLimits(1, 1);
    td::setCacheLimits(64, 256);
    td::reapCompleted();
    // Warm exactly two codes: BGR2HSV (the one that will be left pending) and BGR2YUV.
    cv::triton::cvtColor(src, other, COLOR_BGR2YUV, b);
    cv::triton::cvtColor(src, dst, COLOR_BGR2HSV, a);
    a.waitForCompletion();
    b.waitForCompletion();
    td::reapCompleted();
    // Make this test's own footprint exact: one module slot, one table slot, with BGR2HSV the
    // most recently used entry so the shrink keeps it. The leftover evicted here is not under
    // test, and its release is why the counters are reset afterwards.
    td::setCacheLimits(2, 1);
    td::resetRuntimeStats();

    int ms = 400;
    blockStream(a, &ms);
    const auto t0 = std::chrono::steady_clock::now();
    cv::triton::cvtColor(src, dst, COLOR_BGR2HSV, a);   // module hit and table hit: no load
    const double returnedA =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    td::RuntimeStats st = td::runtimeStats();
    // The pending state is only meaningful if the call did not have to load or compile: either
    // would drain the context and run the block. Asserted, not assumed.
    EXPECT_EQ(0, st.module_loads) << "the measured call loaded a module";
    EXPECT_EQ(0, st.helper_compiles) << "the measured call compiled";
    EXPECT_LT(returnedA, 0.5 * ms) << "the measured call stalled the host, so the launch was "
                                      "not pending and the scenario proves nothing";
    EXPECT_EQ(1, st.launches);
    EXPECT_EQ(1, st.pending);
    const long long freesAtStart = st.module_frees + st.table_frees;
    EXPECT_EQ(0, freesAtStart);

    // Eviction 1, tables: the second code needs its own table set, and the cache holds one.
    // BGR2YUV's module is still cached (two module slots), so no load.
    cv::triton::cvtColor(src, other, COLOR_BGR2YUV, b);
    b.waitForCompletion();
    st = td::runtimeStats();
    EXPECT_EQ(0, st.module_loads) << "the second call loaded a module and drained the context";
    EXPECT_EQ(1, st.table_evictions) << "the pending call's tables were not evicted";
    EXPECT_GE(st.deferred, 1) << "the evicted tables were not counted as held by a pending launch";
    EXPECT_EQ(0, st.table_frees) << "tables a pending launch is reading were freed";

    // Eviction 2, module: the limit shrink evicts the least recently used module, which is the
    // one the pending call is using.
    td::setCacheLimits(1, 1);
    st = td::runtimeStats();
    EXPECT_GE(st.module_evictions, 1) << "no module was evicted";
    EXPECT_GE(st.deferred, 2) << "the module was evicted but not counted as held";
    EXPECT_EQ(0, st.module_frees) << "a module still in use was unloaded";

    // Drain: now the held resources must be released, and the output must be right.
    a.waitForCompletion();
    td::reapCompleted();
    st = td::runtimeStats();
    EXPECT_GE(st.module_frees + st.table_frees, 2)
        << "evicted resources were never released after the launch completed";
    EXPECT_EQ(0, st.pending);
    Mat got;
    dst.download(got);
    EXPECT_EQ(0.0, cvtest::norm(got, want, NORM_INF));
}

// The test table above must cover the module's own code table. Without this, adding a
// conversion to the planner without a test row would pass silently.
TEST(Triton_CvtColor, CoversEverySupportedCode)
{
    const auto& table = kf_cvtcolor::all();
    EXPECT_EQ(sizeof(kCases) / sizeof(kCases[0]), table.size());
    for (const kf_cvtcolor::CodeSpec& spec : table)
    {
        bool found = false;
        for (const CvtCase& c : kCases)
            if (c.code == spec.code) { found = true; break; }
        EXPECT_TRUE(found) << "code " << spec.name << " (" << spec.code << ") has no test row";
    }
}

// Depth inventories must track every admitted planner family, including copies
// whose code names are aliases. Exact counts name numeric rows, not aliases.
TEST(Triton_CvtColor, CoversEverySupportedDepth)
{
    int count16 = 0, count32 = 0;
    for (const auto& spec : kf_cvtcolor::all())
    {
        using Op = kf_cvtcolor::Op;
        const bool admitted16 = spec.op == Op::Bayer || spec.op == Op::Gray ||
            spec.op == Op::Yuv || spec.op == Op::YCrCb ||
            (spec.op == Op::Permute && spec.scn == 1);
        const bool admitted32 = (spec.op == Op::Permute && spec.scn != 2) || spec.op == Op::Gray ||
            spec.op == Op::Yuv || spec.op == Op::YCrCb;
        bool found16 = false, found32 = false;
        for (const auto& c : kCases16U) found16 |= c.code == spec.code;
        for (const auto& c : kCases32F) found32 |= c.code == spec.code;
        EXPECT_EQ(admitted16, found16) << spec.name;
        EXPECT_EQ(admitted32, found32) << spec.name;
        count16 += admitted16; count32 += admitted32;
    }
    EXPECT_EQ(20, count16); EXPECT_EQ(20, count32);
    EXPECT_EQ(size_t(count16), sizeof(kCases16U) / sizeof(kCases16U[0]));
    EXPECT_EQ(size_t(count32), sizeof(kCases32F) / sizeof(kCases32F[0]));
}

// The Xyz/Lab/Luv rows carry coefficients that cvtcolor.cpp computes (coeffs_for) when the table
// is built. This proves the two agree for every such code, so a row added with the wrong `op` or
// a code missing from coeffs_for's lists fails here rather than as a numeric mismatch somewhere
// in a 640x480 image. It also pins the coefficient layout the kernel indexes by: slot order is
// (BLUE-side, green, RED-side), so a BGR code's slot 0 must be the BLUE coefficient and an RGB
// code's slot 0 the RED one -- the exact confusion that produced a wrong matrix on the first
// attempt at this family.
TEST(Triton_CvtColor, CoefficientRowsMatchTheConstructor)
{
    for (const kf_cvtcolor::CodeSpec& spec : kf_cvtcolor::all())
    {
        if (spec.op != kf_cvtcolor::Op::Xyz && spec.op != kf_cvtcolor::Op::Lab &&
            spec.op != kf_cvtcolor::Op::Luv)
            continue;
        int k[9];
        kf_cvtcolor::coeffs_for(spec.code, k);
        for (int i = 0; i < 9; ++i)
            EXPECT_EQ(k[i], spec.k[i]) << spec.name << " coefficient " << i;
    }
    // Spot values with a meaning that can be checked by hand. BGR2XYZ row 0 is 2^12*M[0]/... in
    // the CPU's BGR column order, i.e. {b,g,r} = {0.180423,0.357580,0.412453}*4096 rounded; the
    // BGR2Lab rows are the white-point-scaled sRGB2XYZ matrix, and their sum is exactly 2^12
    // (the CPU's own constructor asserts it, color_lab.cpp:1583).
    const kf_cvtcolor::CodeSpec* x = kf_cvtcolor::find(COLOR_BGR2XYZ);
    ASSERT_TRUE(x != nullptr);
    EXPECT_EQ(739, x->k[0]);
    EXPECT_EQ(1465, x->k[1]);
    EXPECT_EQ(1689, x->k[2]);
    const kf_cvtcolor::CodeSpec* lab = kf_cvtcolor::find(COLOR_BGR2Lab);
    ASSERT_TRUE(lab != nullptr);
    EXPECT_EQ(778, lab->k[0]);
    EXPECT_EQ(1777, lab->k[2]);
    for (int r = 0; r < 3; ++r)
        EXPECT_EQ(4096, lab->k[3 * r] + lab->k[3 * r + 1] + lab->k[3 * r + 2])
            << "RGB2Lab_b row " << r << " must sum to 1 << lab_shift";
    // The reverse matrices are in DESTINATION-SLOT order, so the BGR and RGB variants of one
    // family carry the same rows in the opposite order. Those exact row orders were established
    // by probe-lab/final_layout.cpp, which searches every plausible arrangement and reports the
    // mismatch count against this build's cv::cvtColor -- the winner is 0, the others are 16k+.
    const kf_cvtcolor::CodeSpec* labr = kf_cvtcolor::find(COLOR_Lab2BGR);
    ASSERT_TRUE(labr != nullptr);
    EXPECT_EQ(217, labr->k[0]);      // row 2 of the untransposed matrix: B from the lab triple
    EXPECT_EQ(12615, labr->k[6]);    // row 0: R
    const kf_cvtcolor::CodeSpec* labrrgb = kf_cvtcolor::find(COLOR_Lab2RGB);
    ASSERT_TRUE(labrrgb != nullptr);
    EXPECT_EQ(12615, labrrgb->k[0]); // the RGB destination wants R first, so the rows swap
    EXPECT_EQ(217, labrrgb->k[6]);
    // Luv's reverse matrix is the same XYZ2sRGB matrix without the white-point factor.
    const kf_cvtcolor::CodeSpec* luvr = kf_cvtcolor::find(COLOR_Luv2BGR);
    ASSERT_TRUE(luvr != nullptr);
    EXPECT_EQ(228, luvr->k[0]);
    EXPECT_EQ(13273, luvr->k[6]);
}

}}  // namespace
