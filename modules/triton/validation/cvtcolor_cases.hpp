// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_TRITON_VALIDATION_CVTCOLOR_CASES_HPP
#define OPENCV_TRITON_VALIDATION_CVTCOLOR_CASES_HPP
#include "opencv2/imgproc.hpp"
namespace opencv_test { namespace triton_test { namespace cvt_cases {
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
// caught by test_cvtcolor.cpp::CoversEverySupportedCode.
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

}}}
#endif
