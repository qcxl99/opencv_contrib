// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Pure part of the cvtColor provider: the code table, request validation and the explicit
// compile key. No device access, so it is testable without a GPU (tests/test_cvtcolor_plan.py).
//
// The supported set is the conversions that OpenCV computes with INTEGER arithmetic on CV_8U.
// Not because "integer" alone implies two implementations agree -- it does not -- but because
// integer arithmetic makes each output pixel a function of its own input bytes and of constant
// tables only: no reduction, no accumulation order, no SIMD width, hence no position
// dependence within a row. That is verifiable, and it was verified together with equality to
// this tree's cv::cvtColor over the whole 8-bit input space (runs/cvtcolor-module-0001).
// "Exact" below therefore means exact against that reference; a build that routes cvtColor
// through IPP, a custom HAL, or a changed coefficient set is a different reference and must be
// re-verified (test/test_cvtcolor.cpp refuses to run on an IPP build for that reason).
//
//   BGR<->RGB, BGR<->BGRA, RGB<->RGBA, BGRA<->RGBA        pure channel gathers
//   BGR/RGB -> GRAY and BGRA/RGBA -> GRAY                 CV_DESCALE, gray_shift = 15
//   BGR/RGB <-> YUV, BGR/RGB <-> YCrCb                    fixed point, yuv_shift = 14
//   BGR/RGB -> HSV                                        integer, hsv_shift = 12
//
// DELIBERATELY REFUSED, with the reason recorded rather than a tolerance invented:
//   * HSV2BGR, and every HLS code. OpenCV computes these in float32, and the SAME build
//     rounds differently in its SIMD body than in its scalar tail (the body truncates the
//     final *255, the tail rounds). One pixel's answer therefore depends on where it sits in
//     the row, so there is no single CPU result to match.
//   * Everything outside CV_8U (float32, CV_16U), 2-channel images, and the codes whose
//     CPU path is a LUT or an integral image rather than a per-pixel formula.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "opencv2/core/cvdef.h"   // CV_EXPORTS
#include "provider.hpp"

namespace kf_cvtcolor {

// Operation ids. These are the OP constexprs of kernels/cvtcolor_cpu.py.
enum class Op : int { Gray = 0, Permute = 1, Yuv = 2, YCrCb = 3, Hsv = 4 };

struct CodeSpec {
    const char* name;      // the OpenCV constant name, e.g. "BGR2YUV"
    int code;              // the numeric cv::COLOR_* value
    Op op;
    int scn, dcn;          // source / destination channel count
    int blue;              // blue index: 0 for a BGR-order source, 2 for an RGB-order one
    bool inverse;          // YUV/YCrCb direction; ignored by the other ops
    int hrange;            // Hsv only: 180 for HSV, 256 for HSV_FULL (h += h<0 ? hrange : 0)
    int perm[4];           // Permute only: source channel for each dst channel, -1 = 255
};

// The table. A code that is not here is refused by name; there is no nearest-neighbour
// substitution and no fallback to the CPU implementation.
const CodeSpec* find(int code);
// CV_EXPORTS because OpenCV builds with hidden visibility and the accuracy test reads this
// table to prove the test's own coverage (tests/test_cvtcolor.cpp, CoversEverySupportedCode).
CV_EXPORTS const std::vector<CodeSpec>& all();

struct Request {
    kf_resize_rt::DeviceView src, dst;
    int code = 0;
};

struct Normalized {
    const CodeSpec* spec = nullptr;
    int rows = 0, cols = 0;
    long long n = 0;            // output pixels = rows * cols
};

// Pure checks. On success `out.spec` points into the static table.
kf_resize_rt::Status normalize(const Request& req, Normalized& out);

// The two int[256] tables RGB2HSV_b divides by, built exactly as the CPU builds them
// (color_hsv.simd.hpp: hsv_shift = 12, round-to-nearest on a double division).
struct HsvTables {
    std::vector<int> sdiv, hdiv;
};
HsvTables hsv_tables();
// The same, for the _FULL codes: identical sdiv, hdiv built from (256 << 12) instead of
// (180 << 12), i.e. a hue range of 0..255.
HsvTables hsv_tables_full();

// The runtime-argument list of the one jit function, in Triton's type strings. Constexpr
// parameters are not runtime arguments. tests/test_cvtcolor_plan.py compares this with the
// signature the JIT actually compiles.
std::string signature(const Normalized& n);

// The explicit compile key. Triton is given no implicit specialisation (every channel count
// and the operation are constexprs, every other argument is do_not_specialize), so this key
// is complete: equal keys run the same binary whatever the sizes or pointers.
std::string constexprs_text(const Normalized& n);
int block_size(const Normalized& n);
int num_warps(const Normalized& n);
void grid(const Normalized& n, int block, unsigned out[3]);   // {rows, ceil(cols/BLOCK), 1}

}  // namespace kf_cvtcolor
