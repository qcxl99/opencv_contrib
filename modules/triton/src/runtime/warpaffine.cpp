// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "warpaffine.hpp"

#include "opencv2/core/utility.hpp"   // cv::useOptimized
#include "opencv2/imgproc.hpp"        // cv::INTER_AREA (the CPU rewrite this follows)

#include <algorithm>
#include <cmath>
#include <sstream>
#include <map>
#include <mutex>
#include <tuple>

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

// Hardware FMA support does not identify the arithmetic compiled into imgproc.
// Calibrate the linked CPU vector path at rounding ties, with no CPU feature
// toggles and no device work. Padding keeps every sampled column out of the tail.
// Type/channel dispatch and useOptimized are part of the calibration identity.
bool reference_fused(int depth, int channels)
{
    static std::mutex mutex;
    static std::map<std::tuple<int, int, bool>, bool> cache;
    const auto key = std::make_tuple(depth, channels, cv::useOptimized());
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = cache.find(key);
    if (found != cache.end()) return found->second;

    cv::Mat ids(37, 53, CV_MAKETYPE(CV_16U, channels));
    for (int y = 0; y < ids.rows; ++y)
        for (int x = 0; x < ids.cols; ++x)
            for (int c = 0; c < channels; ++c)
                ids.ptr<unsigned short>(y)[x * channels + c] =
                    static_cast<unsigned short>(1 + (y * ids.cols + x) * 17 + c * 73);
    cv::Mat source;
    ids.convertTo(source, CV_MAKETYPE(depth, channels));
    if (depth == CV_8U)
        for (int y = 0; y < source.rows; ++y)
            for (int x = 0; x < source.cols * channels; ++x)
                source.ptr<unsigned char>(y)[x] = static_cast<unsigned char>(ids.ptr<unsigned short>(y)[x] % 251 + 1);

    // Fixed probes are independent of the user's matrix, seed or expected image.
    const double probes[][6] = {{1, 1.9, -20, .4, 1, -10},
                                {.9, .43589, -14, -.43589, .9, 10}};
    bool matches[2] = {true, true};
    for (const auto& matrix : probes)
    {
        double sampling[6];
        invert(matrix, sampling);
        float a[6];
        for (int i = 0; i < 6; ++i) a[i] = static_cast<float>(sampling[i]);
        cv::Mat reference;
        cv::warpAffine(source, reference, cv::Mat(2, 3, CV_64F, const_cast<double*>(matrix)),
                       cv::Size(256, 37), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
        cv::Mat src64, ref64;
        source.convertTo(src64, CV_MAKETYPE(CV_64F, channels));
        reference.convertTo(ref64, CV_MAKETYPE(CV_64F, channels));
        for (int fused = 0; fused < 2; ++fused)
            for (int y = 0; y < 37 && matches[fused]; ++y)
                for (int x = 0; x < 53 && matches[fused]; ++x)
                {
                    // Volatile intermediates prohibit compiler FMA contraction
                    // in the unfused alternative, regardless of module flags.
                    volatile float rowx = y * a[1], rowy = y * a[4];
                    const float rx = rowx + a[2], ry = rowy + a[5];
                    volatile float colx = x * a[0], coly = x * a[3];
                    const float sx = fused ? std::fma(float(x), a[0], rx) : colx + rx;
                    const float sy = fused ? std::fma(float(x), a[3], ry) : coly + ry;
                    const int ix = cvRound(sx), iy = cvRound(sy);
                    for (int c = 0; c < channels; ++c)
                    {
                        const double predicted = ix >= 0 && ix < 53 && iy >= 0 && iy < 37
                            ? src64.ptr<double>(iy)[ix * channels + c] : 0;
                        if (predicted != ref64.ptr<double>(y)[x * channels + c])
                            matches[fused] = false;
                    }
                }
    }
    if (matches[0] == matches[1])
        CV_Error(cv::Error::StsUnsupportedFormat,
                 "cv::triton::warpAffine: linked CPU vector arithmetic could not be identified");
    cache.emplace(key, matches[1]);
    return matches[1];
}

bool is_border(int b)
{
    return b == 0 || b == 1 || b == 2 || b == 3 || b == 4 || b == 5;
    // CONSTANT/REPLICATE/REFLECT/WRAP/REFLECT_101/TRANSPARENT. TRANSPARENT preserves outside
    // nearest samples; LINEAR/AREA use the old destination for each outside tap.
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

    // INTER_AREA is not an approximation here: cv::warpAffine rewrites it to INTER_LINEAR before
    // doing anything else (imgwarp.cpp, `if (interpolation == INTER_AREA) interpolation =
    // INTER_LINEAR;`). Measured on the reference build: INTER_AREA and INTER_LINEAR produce
    // byte-identical output for the same matrix and size. So the rewrite is followed rather than
    // refused -- refusing it made the module reject a call the CPU accepts and answers exactly.
    // INTER_LANCZOS4 stays refused: the CPU has no rewrite for it and its result is not the same
    // computation.
    const int interp = req.interpolation == cv::INTER_AREA ? cv::INTER_LINEAR : req.interpolation;
    if (interp != 0 && interp != 1 && interp != 2)
        return unsupported(Reason::Interpolation,
                           "cv::triton::warpAffine: interpolation " +
                               std::to_string(req.interpolation) +
                               " is not served by this module (warpaffine/0.4.3 covers "
                               "INTER_NEAREST, INTER_LINEAR, INTER_CUBIC and INTER_AREA); "
                               "INTER_LANCZOS4 is refused rather than approximated");
    // BORDER_TRANSPARENT is served for NEAREST and LINEAR (including the AREA alias).
    // TransparentMatchesCpu checks the selected reference with initialized destinations,
    // every supported dtype/channel, ROI guards and inside/partial/outside geometries.
    // CUBIC + TRANSPARENT remains outside the implemented subset.
    if (req.border_mode == cv::BORDER_TRANSPARENT && req.interpolation == cv::INTER_CUBIC)
        return unsupported(Reason::Interpolation,
                           "cv::triton::warpAffine: BORDER_TRANSPARENT with INTER_CUBIC is not "
                           "served (its destination-fill blend is not reproduced); TRANSPARENT "
                           "works with INTER_NEAREST, INTER_LINEAR and INTER_AREA");
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
    // 2-channel and CV_16S are legal cv::warpAffine inputs but are NOT served here yet, and the
    // reason is measured rather than assumed: cv::warpAffine does not use one arithmetic for all
    // types. genericWarp's getImgWarpFunc (imgwarp.cpp:70) returns non-null only for
    // INTER_CUBIC, so NEAREST/LINEAR fall through to hal::warpAffine (imgwarp.cpp:2272), whose
    // switches list C1/C3/C4 of 8U, 16U and 32F -- "// no default" after each. C2 and every CV_16S
    // type therefore run the legacy fixed-point WarpAffineInvoker, not the float invoker this
    // kernel reproduces. Measured: a CV_8UC2 warp differs from the same data warped as two
    // CV_8UC1 planes by up to 6 levels, while CV_8UC3 matches the split-C1 control exactly (0).
    // Serving them from the float model would be silently wrong, so they are refused by name.
    if (s.channels != 1 && s.channels != 3 && s.channels != 4)
        return unsupported(Reason::DepthOrChannels,
                           "cv::triton::warpAffine: channels must be 1, 3 or 4; 2-channel data is "
                           "accepted by cv::warpAffine but runs its fixed-point path, which this "
                           "module does not reproduce yet");
    if (!(s.ctx == d.ctx)) return unsupported(Reason::ContextMismatch, "src and dst belong to different contexts");
    Status valid = kf_resize_rt::validate_view(s, "src");
    if (!valid.ok()) return valid;
    valid = kf_resize_rt::validate_view(d, "dst");
    if (!valid.ok()) return valid;
    if (s.ptr < d.end() && d.ptr < s.end())
        return unsupported(Reason::InPlaceOrOverlap, "src and dst overlap (in-place is not supported)");

    out.mode = interp == 0 ? Mode::Nearest : (interp == 1 ? Mode::Linear : Mode::Cubic);
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
    //
    // The tile SHAPE measured faster for the interpolating modes on a rotating geometry, and the
    // likely reason -- a hypothesis, the causal link was not profiled -- is that a rotation maps
    // an output row onto a diagonal: over 128 lanes of one row the source row moves by about
    // 0.44*128 ~= 56 rows, so one warp's request scatters across ~56 source rows. A 2x64 tile
    // covers the same 128 pixels with half that vertical spread. The timings below are measured
    // (one SM-90 device, same process, median of graph replays):
    //
    //   CUBIC  1920x1080 98.2 -> 84.9 us (1.16x), 3840x2160 399 -> 341 us (1.17x)
    //   rotate 1920x1080 29.5 -> 26.9 us (1.10x), 3840x2160 114 -> 104 us (1.10x)
    //   linear (identity map) unchanged at 26.0 / 103 us
    //
    // NEAREST measured the opposite preference: 1x256 gave 8.9/32.1 us against 10.9/45.5 for
    // 1x128 (1.23x at 1080p, 1.42x at 4K), and using the interpolation shape there cost 8%.
    // One reading consistent with both results is the tap count -- 1 tap rather than 16 -- but
    // that is an interpretation, not something this measurement separates from other causes.
    if (out.mode == Mode::Nearest) {
        out.block_h = 1;
        out.block_w = 256;
    } else {
        out.block_h = 2;
        out.block_w = 64;
    }
    out.num_warps = 4;

    const int depth = out.u8 ? CV_8U : (out.u16 ? CV_16U : CV_32F);
    out.fused = reference_fused(depth, s.channels);
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
    o << "BLOCK_H=" << n.block_h << ",BLOCK_W=" << n.block_w << ",BORDER=" << n.border_mode_code
      << ",BORDER_X=" << n.border_x
      << ",BORDER_Y=" << n.border_y;
    // U8/U16 exist only on the kernels that convert their loads (LINEAR, CUBIC); the NEAREST
    // kernel copies the element unchanged, so its dtype is carried by the pointer type alone
    // and declaring an unused constexpr would make the compile helper's key check fail.
    if (n.mode != Mode::Nearest)
        o << ",U8=" << (n.u8 ? 1 : 0) << ",U16=" << (n.u16 ? 1 : 0);
    o << ",FUSED=" << (n.fused ? 1 : 0);
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
