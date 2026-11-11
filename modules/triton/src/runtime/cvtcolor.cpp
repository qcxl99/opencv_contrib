// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Pure part of the cvtColor provider: the code table, request validation and the explicit
// compile key. No device access, so it is testable without a GPU.
//
// Why this subset, and what "exact" is exact against. The reference is THIS build's
// cv::cvtColor, not "the CPU" in the abstract: integer arithmetic does not by itself make two
// implementations produce the same bytes, and cv::cvtColor does dispatch through
// cv_hal_cvtBGRtoBGR / cvtBGRtoYUV / cvtBGRtoHSV, so a HAL build is a different reference.
// What integer arithmetic does buy here is stronger and checkable: each output pixel depends
// only on its own input bytes and on 256-entry constant tables, so the result cannot depend on
// where the pixel sits in a row -- no reduction, no accumulation order, no SIMD width. That is
// the property that makes the result reproducible, and it was measured (not assumed), together
// with equality to the reference over the whole 8-bit input space, in
// runs/cvtcolor-module-0001 (exhaustive.cpp; identity.json records the exact sources):
//
//   channel permutes                    pure gathers
//   BGR/RGB/BGRA/RGBA -> GRAY           CV_DESCALE, gray_shift = 15
//   BGR/RGB <-> YUV, YCrCb              fixed point, yuv_shift = 14
//   BGR/RGB -> HSV, HSV_FULL            integer, hsv_shift = 12, 180 or 256 hue range
//
// Scope: the claim holds for the build recorded in runs/cvtcolor-module-0001/identity.json --
// generic C++ path, no IPP, no custom HAL -- and is re-checked there by
// test_cvtcolor.cpp's ReferenceBuildIsCharacterised, which reads those fields by value and
// fails if they change. An earlier version of this comment claimed color.cpp had no HAL entry
// points; it does, one level down in color_rgb/color_yuv/color_hsv.dispatch.cpp, which is why
// the reference is a specific build rather than "the CPU".
//
// Refused, with the reason recorded instead of a tolerance invented:
//   * HSV2BGR and every HLS code. OpenCV computes those in float32, and ONE BUILD rounds
//     differently in its SIMD body than in its scalar tail (the body truncates the final
//     *255, the tail rounds to nearest). Measured on a live build: the same HSV pixel
//     converts differently depending on where it sits in the row. There is no single CPU
//     answer to match, so claiming "bit-exact" would be claiming something undefined.
//   * Anything outside CV_8U, 2-channel images, and any code not in the table below.
#include "cvtcolor.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace kf_cvtcolor {

using kf_resize_rt::Code;
using kf_resize_rt::Reason;
using kf_resize_rt::Status;

namespace {

// perm[]: source channel for each destination channel; -1 means "write 255".
const CodeSpec TABLE[] = {
    // ---- channel permutes (scn, dcn, blue, inverse, perm)
    {"BGR2BGRA",     0,  Op::Permute, 3, 4, 0, false, 180, {0, 1, 2, -1}},
    {"BGRA2BGR",     1,  Op::Permute, 4, 3, 0, false, 180, {0, 1, 2, -1}},
    {"BGR2RGBA",     2,  Op::Permute, 3, 4, 0, false, 180, {2, 1, 0, -1}},
    {"RGBA2BGR",     3,  Op::Permute, 4, 3, 0, false, 180, {2, 1, 0, -1}},
    {"BGR2RGB",      4,  Op::Permute, 3, 3, 0, false, 180, {2, 1, 0, -1}},
    {"BGRA2RGBA",    5,  Op::Permute, 4, 4, 0, false, 180, {2, 1, 0, 3}},
    // ---- to gray (blue index selects channel order; OpenCV accepts both)
    {"BGR2GRAY",     6,  Op::Gray, 3, 1, 0, false, 180, {0, 0, 0, 0}},
    {"RGB2GRAY",     7,  Op::Gray, 3, 1, 2, false, 180, {0, 0, 0, 0}},
    {"BGRA2GRAY",    10, Op::Gray, 4, 1, 0, false, 180, {0, 0, 0, 0}},
    {"RGBA2GRAY",    11, Op::Gray, 4, 1, 2, false, 180, {0, 0, 0, 0}},
    // ---- YCrCb
    {"BGR2YCrCb",    36, Op::YCrCb, 3, 3, 0, false, 180, {0, 0, 0, 0}},
    {"RGB2YCrCb",    37, Op::YCrCb, 3, 3, 2, false, 180, {0, 0, 0, 0}},
    {"YCrCb2BGR",    38, Op::YCrCb, 3, 3, 0, true,  180, {0, 0, 0, 0}},
    {"YCrCb2RGB",    39, Op::YCrCb, 3, 3, 2, true,  180, {0, 0, 0, 0}},
    // ---- HSV, integer path, hue range 0..180
    {"BGR2HSV",      40, Op::Hsv, 3, 3, 0, false, 180, {0, 0, 0, 0}},
    {"RGB2HSV",      41, Op::Hsv, 3, 3, 2, false, 180, {0, 0, 0, 0}},
    // ---- HSV, full hue range 0..255: same code, a different div table
    {"BGR2HSV_FULL", 66, Op::Hsv, 3, 3, 0, false, 256, {0, 0, 0, 0}},
    {"RGB2HSV_FULL", 67, Op::Hsv, 3, 3, 2, false, 256, {0, 0, 0, 0}},
    // ---- YUV
    {"BGR2YUV",      82, Op::Yuv, 3, 3, 0, false, 180, {0, 0, 0, 0}},
    {"RGB2YUV",      83, Op::Yuv, 3, 3, 2, false, 180, {0, 0, 0, 0}},
    {"YUV2BGR",      84, Op::Yuv, 3, 3, 0, true,  180, {0, 0, 0, 0}},
    {"YUV2RGB",      85, Op::Yuv, 3, 3, 2, true,  180, {0, 0, 0, 0}},
};

Status prepared() {
    Status s;
    s.code = Code::Prepared;
    return s;
}

Status unsupported(Reason r, std::string msg) {
    Status s;
    s.code = Code::Unsupported;
    s.reason = r;
    s.message = std::move(msg);
    return s;
}

}  // namespace

const CodeSpec* find(int code) {
    for (const CodeSpec& c : TABLE)
        if (c.code == code) return &c;
    return nullptr;
}

const std::vector<CodeSpec>& all() {
    static const std::vector<CodeSpec> v(TABLE, TABLE + sizeof(TABLE) / sizeof(TABLE[0]));
    return v;
}

Status normalize(const Request& req, Normalized& out) {
    const kf_resize_rt::DeviceView &s = req.src, &d = req.dst;
    const CodeSpec* spec = find(req.code);
    if (!spec)
        return unsupported(Reason::Interpolation,
                           "cvtColor code " + std::to_string(req.code) +
                               " is not served by this module: see cvtcolor.hpp for the "
                               "supported set and why the float32 conversions are excluded");
    if (s.type != kf_resize_rt::ElemType::U8 || d.type != kf_resize_rt::ElemType::U8)
        return unsupported(Reason::DepthOrChannels, "cvtColor here covers CV_8U only");
    if (s.channels != spec->scn)
        return unsupported(Reason::DepthOrChannels,
                           std::string(spec->name) + " needs a " + std::to_string(spec->scn) +
                               "-channel source, got " + std::to_string(s.channels));
    if (d.channels != spec->dcn)
        return unsupported(Reason::DepthOrChannels,
                           std::string(spec->name) + " writes " + std::to_string(spec->dcn) +
                               " channels, dst has " + std::to_string(d.channels));
    if (s.rows != d.rows || s.cols != d.cols)
        return unsupported(Reason::EmptyOrSize, "cvtColor does not change the image size");
    if (!(s.ctx == d.ctx)) return unsupported(Reason::ContextMismatch, "src and dst belong to different contexts");
    if (s.ptr == 0 || d.ptr == 0 || s.rows <= 0 || s.cols <= 0)
        return unsupported(Reason::EmptyOrSize, "src or dst is empty");
    for (const char* which : {"src", "dst"}) {
        const kf_resize_rt::DeviceView& v = which[0] == 's' ? s : d;
        if (v.step % kf_resize_rt::elem_size(v.type) != 0 || v.step < v.row_bytes())
            return unsupported(Reason::StepLayout, std::string(which) +
                               ": step must be a multiple of the element size and >= the row bytes");
    }
    if (s.ptr < d.end() && d.ptr < s.end())
        return unsupported(Reason::InPlaceOrOverlap, "src and dst overlap (in-place is not supported)");
    const long long n = static_cast<long long>(s.rows) * s.cols;
    if (n > 0x7fffffffLL)
        return unsupported(Reason::IndexRange, "pixel count exceeds int32 indexing");
    out.spec = spec;
    out.rows = s.rows;
    out.cols = s.cols;
    out.n = n;
    return prepared();
}

HsvTables hsv_tables() {
    // color_hsv.simd.hpp: hsv_shift = 12; sdiv = (255<<12)/i, hdiv = (range<<12)/(6*i),
    // both saturate_cast<int> of a double division, i = 1..255, entry 0 is 0.
    HsvTables t;
    t.sdiv.resize(256, 0);
    t.hdiv.resize(256, 0);
    for (int i = 1; i < 256; ++i) {
        t.sdiv[i] = static_cast<int>(std::lrint(1044480.0 / i));          // (255 << 12) / i
        t.hdiv[i] = static_cast<int>(std::lrint(737280.0 / (6.0 * i)));  // (180 << 12) / (6 i)
    }
    return t;
}

HsvTables hsv_tables_full() {
    // hdiv for the 0..255 hue range: (256 << 12) / (6 i).
    HsvTables t;
    t.sdiv.resize(256, 0);
    t.hdiv.resize(256, 0);
    for (int i = 1; i < 256; ++i) {
        t.sdiv[i] = static_cast<int>(std::lrint(1044480.0 / i));
        t.hdiv[i] = static_cast<int>(std::lrint(1048576.0 / (6.0 * i)));  // (256 << 12) / (6 i)
    }
    return t;
}

std::string signature(const Normalized& n) {
    std::ostringstream o;
    // Order matters: this must be the kernel's own parameter order
    // (_cvtcolor_cpu_kernel(src, dst, cols, src_rs, dst_rs, sdiv, hdiv, ...)), and the compile
    // helper rejects a signature that disagrees with it.
    //
    // cols and the two row strides are runtime arguments rather than constexprs on purpose:
    // making them constexpr would compile a separate binary for every image width. They are
    // i32, and the two strides are in BYTES (DeviceView::step), which is also the addressing
    // unit the kernel uses -- U8 only, so a pixel is SCN bytes and `row*rs + col*SCN` is
    // already a byte offset.
    (void)n;
    o << "src:*u8, dst:*u8, cols:i32, src_rs:i32, dst_rs:i32, sdiv:*i32, hdiv:*i32";
    return o.str();
}

std::string constexprs_text(const Normalized& n) {
    const CodeSpec& s = *n.spec;
    std::ostringstream o;
    o << "BLOCK=" << block_size(n) << ",BLUE=" << s.blue << ",DCN=" << s.dcn
      << ",I0=" << s.perm[0] << ",I1=" << s.perm[1] << ",I2=" << s.perm[2] << ",I3=" << s.perm[3]
      << ",HR=" << s.hrange << ",INV=" << (s.inverse ? 1 : 0) << ",OP=" << static_cast<int>(s.op)
      << ",SCN=" << s.scn;
    return o.str();
}

int block_size(const Normalized& n) {
    // Output pixels per program along the row. A tuning constant: the grid is (rows,
    // ceil(cols/BLOCK)), so this is the only shape the kernel depends on and it stays fixed
    // to keep one binary per (SCN, DCN, OP, ...) rather than one per image size. 256 covers a
    // full row of most images in one program without spilling registers in the 3-channel
    // branches.
    (void)n;
    return 256;
}

int num_warps(const Normalized& n) {
    (void)n;
    return 4;
}

void grid(const Normalized& n, int block, unsigned out[3]) {
    // One program per (row, column block). Rows go on x because y is limited to 65535 on
    // every supported SM generation and y would be the unbounded axis for a tall image.
    const long long b = block > 0 ? block : 256;
    out[0] = static_cast<unsigned>(n.rows);
    out[1] = static_cast<unsigned>((n.cols + b - 1) / b);
    out[2] = 1;
}

}  // namespace kf_cvtcolor
