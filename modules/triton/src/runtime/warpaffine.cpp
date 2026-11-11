// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "warpaffine.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace kf_warpaffine {

using kf_resize_rt::Reason;
using kf_resize_rt::Status;

namespace {
Status prepared() {
    Status s;
    s.code = kf_resize_rt::Code::Prepared;
    return s;
}

Status unsupported(Reason r, std::string msg) {
    Status s;
    s.code = kf_resize_rt::Code::Unsupported;
    s.reason = r;
    s.message = std::move(msg);
    return s;
}

bool is_border(int b)
{
    return b == 0 || b == 1 || b == 2 || b == 3 || b == 4;   // CONSTANT/REPLICATE/REFLECT/WRAP/REFLECT_101
}
}  // namespace

void invert(const double M[6], double out[6])
{
    double D = M[0] * M[4] - M[1] * M[3];
    D = D != 0 ? 1.0 / D : 0.0;
    const double a11 = M[4] * D, a22 = M[0] * D;
    out[0] = a11;
    out[1] = M[1] * -D;
    out[3] = M[3] * -D;
    out[4] = a22;
    out[2] = -out[0] * M[2] - out[1] * M[5];
    out[5] = -out[3] * M[2] - out[4] * M[5];
}

Status normalize(const Request& req, Normalized& out)
{
    const kf_resize_rt::DeviceView &s = req.src, &d = req.dst;

    if (req.interpolation != 0 && req.interpolation != 1 && req.interpolation != 2)
        return unsupported(Reason::Interpolation,
                           "cv::triton::warpAffine: interpolation " +
                               std::to_string(req.interpolation) +
                               " is not served by this module (warpaffine/0.4.1 covers "
                               "INTER_NEAREST, INTER_LINEAR and INTER_CUBIC); INTER_LANCZOS4 "
                               "and INTER_AREA are refused rather than approximated");
    if (!is_border(req.border_mode))
        return unsupported(Reason::Interpolation,
                           "cv::triton::warpAffine: border mode " + std::to_string(req.border_mode) +
                               " is not one of CONSTANT/REPLICATE/REFLECT/WRAP/REFLECT_101");
    if (s.type != d.type || s.channels != d.channels)
        return unsupported(Reason::TypeMismatch, "src and dst must have the same depth and channel count");
    if (s.type != kf_resize_rt::ElemType::U8 && s.type != kf_resize_rt::ElemType::U16 &&
        s.type != kf_resize_rt::ElemType::F32)
        return unsupported(Reason::DepthOrChannels,
                           "cv::triton::warpAffine: CV_8U, CV_16U and CV_32F only");
    if (s.channels != 1 && s.channels != 3 && s.channels != 4)
        return unsupported(Reason::DepthOrChannels, "channels must be 1, 3 or 4");
    if (!(s.ctx == d.ctx)) return unsupported(Reason::ContextMismatch, "src and dst belong to different contexts");
    if (s.ptr == 0 || d.ptr == 0 || s.rows <= 0 || s.cols <= 0 || d.rows <= 0 || d.cols <= 0)
        return unsupported(Reason::EmptyOrSize, "src or dst is empty");
    for (const char* which : {"src", "dst"}) {
        const kf_resize_rt::DeviceView& v = which[0] == 's' ? s : d;
        if (v.step % kf_resize_rt::elem_size(v.type) != 0 || v.step < v.row_bytes())
            return unsupported(Reason::StepLayout, std::string(which) +
                               ": step must be a multiple of the element size and >= the row bytes");
    }
    const long long n = static_cast<long long>(d.rows) * d.cols * d.channels;
    if (n > 0x7fffffffLL)
        return unsupported(Reason::IndexRange, "output element count exceeds int32 indexing");
    if (s.ptr < d.end() && d.ptr < s.end())
        return unsupported(Reason::InPlaceOrOverlap, "src and dst overlap (in-place is not supported)");

    out.mode = req.interpolation == 0 ? Mode::Nearest
             : (req.interpolation == 1 ? Mode::Linear : Mode::Cubic);
    out.u8 = s.type == kf_resize_rt::ElemType::U8;
    out.u16 = s.type == kf_resize_rt::ElemType::U16;
    out.src_h = s.rows;
    out.src_w = s.cols;
    out.dst_h = d.rows;
    out.dst_w = d.cols;
    out.channels = s.channels;
    if (req.inverse_map)
        std::copy(req.M, req.M + 6, out.A);
    else
        invert(req.M, out.A);

    out.border_mode_code = req.border_mode;
    out.border_x = req.border_mode;
    out.border_y = req.border_mode;
    if (req.border_mode != 0) {          // not BORDER_CONSTANT: a 1-pixel axis replicates
        if (out.src_w <= 1) out.border_x = 1;
        if (out.src_h <= 1) out.border_y = 1;
    }
    for (int i = 0; i < 4; ++i) out.border[i] = req.border[i];

    // One tile of output pixels per program, the same knob shape resize and cvtColor expose.
    // BLOCK_W 128 with BLOCK_H 1 measured best on the qualification set
    // (runs/warpaffine-0001); the search can still override it through LaunchConfig.
    out.block_h = 1;
    out.block_w = 128;
    out.num_warps = 4;
    return prepared();
}

std::string signature(const Normalized& n)
{
    // Must match the kernel's own parameter order:
    //   (src, dst, M00..M12, src_h, src_w, dst_h, dst_w, <constexprs>)
    // The six matrix entries are fp32 runtime arguments: making them constexpr would compile a
    // separate binary per affine matrix.
    // The four border components are RUNTIME fp32 arguments, never constexprs: as constexprs
    // they would be truncated to integers, which is wrong for CV_32F (a border of 0.5), and
    // they would also bake every border value into a separate compiled binary. fp32 rather
    // than the source dtype because a u8-typed scalar argument is not a legal ABI for this
    // kernel; the kernel converts to its own dtype, which is the same saturate/round the CPU
    // does.
    // The pointer types are concrete per dtype ("*u8" / "*fp32"), not a placeholder: the
    // compile helper maps each token to a Triton type and has no generic form.
    const char* t = n.u8 ? "u8" : (n.u16 ? "u16" : "fp32");
    std::ostringstream o;
    o << "src:*" << t << ", dst:*" << t
      << ", M00:fp32, M01:fp32, M02:fp32, M10:fp32, M11:fp32, M12:fp32, "
         "src_h:i32, src_w:i32, dst_h:i32, dst_w:i32, src_rs:i32, dst_rs:i32, "
         "bv0:fp32, bv1:fp32, bv2:fp32, bv3:fp32";
    return o.str();
}

std::string constexprs_text(const Normalized& n)
{
    std::ostringstream o;
    o << "BLOCK_H=" << n.block_h << ",BLOCK_W=" << n.block_w << ",BORDER=" << n.border
      << ",BORDER_X=" << n.border_x
      << ",BORDER_Y=" << n.border_y;
    // U8/U16 exist only on the kernels that convert their loads (LINEAR, CUBIC); the NEAREST
    // kernel copies the element unchanged, so its dtype is carried by the pointer type alone
    // and declaring an unused constexpr would make the compile helper's key check fail.
    if (n.mode != Mode::Nearest)
        o << ",U8=" << (n.u8 ? 1 : 0) << ",U16=" << (n.u16 ? 1 : 0);
    o << ",cn=" << n.channels;
    return o.str();
}

void grid(const Normalized& n, unsigned out[3])
{
    const int bh = n.block_h > 0 ? n.block_h : 1;
    const int bw = n.block_w > 0 ? n.block_w : 128;
    out[0] = static_cast<unsigned>((n.dst_h + bh - 1) / bh);
    out[1] = static_cast<unsigned>((n.dst_w + bw - 1) / bw);
    out[2] = 1;
}

}  // namespace kf_warpaffine
