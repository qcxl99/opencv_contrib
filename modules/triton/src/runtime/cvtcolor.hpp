// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Pure code/depth admission, request validation and explicit compile keys.
// No device access: CPU controls can exercise normalize without launching work.
//
// all() contains 84 numeric rows; enum aliases do not add rows. CV_8U serves
// every row. CV_16U serves 20: Bayer, colour/gray and YUV/YCrCb. CV_32F serves
// 20: channel permutations and gray copies, Op::Gray,
// Op::Yuv and Op::YCrCb. The two 2-channel Y extractions are 8U-only,
// as required by the CPU cvtColor wrapper. Integer conversions and copies use exact comparison;
// float arithmetic uses 8*2^-24*max(1,|ref|,max|input channels at this pixel|).
//
// Two-channel images are admitted only by explicit rows: 565/555 packed bytes,
// 422 decode and Y-channel extraction. The six 422 conversions require even
// width, while extraction allows odd width. 420 decode takes a packed 3H/2 by
// W image and produces H by W. normalize checks these geometry rules before
// any compile/launch and reuses the provider's shared DeviceView validation.
//
// Unsupported depths/codes (reverse HSV, HLS, forward linear Luv, Bayer VNG/EA,
// YUV encoding) are refused; no nearby conversion or CPU fallback is selected.
// Historic CPU formula characterization in runs/cvtcolor-ext-0001 informs the
// numerical criterion, but cannot by itself certify a GPU module or a later
// build. test/test_cvtcolor.cpp exercises this module's actual public GPU entry
// against that build's own cv::cvtColor; changed IPP/HAL/toolchains need their
// own evidence. The module README owns the public scope and limitations.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "opencv2/core/cvdef.h"   // CV_EXPORTS
#include "provider.hpp"

namespace kf_cvtcolor {

// Operation ids. These are the OP constexprs of kernels/cvtcolor_cpu.py.
enum class Op : int {
    Gray = 0, Permute = 1, Yuv = 2, YCrCb = 3, Hsv = 4, Packed = 5, Bayer = 6,
    Xyz = 7, Lab = 8, Luv = 9, Yuv422 = 10, Yuv420 = 11
};

struct CodeSpec {
    const char* name;      // the OpenCV constant name, e.g. "BGR2YUV"
    int code;              // the numeric cv::COLOR_* value
    Op op;
    int scn, dcn;          // source / destination channel count
    int blue;              // blue index: 0 for a BGR-order source, 2 for an RGB-order one
    bool inverse;          // direction: YUV/YCrCb inverse, and Packed decode (2ch -> colour)
    int hrange;            // Hsv only: 180 for HSV, 256 for HSV_FULL (h += h<0 ? hrange : 0)
    int perm[4];           // Permute only: source channel for each dst channel, -1 = 255
    // Packed only: green-bit count, the CPU's own discriminator between the two formats
    // (color.hpp greenBits(): 6 for every BGR565 code, 5 for every BGR555 code). It is a
    // field rather than something derived from the code value because the CPU decides by
    // passing 6 or 5 down to the invoker -- the code number is not consulted there.
    int gb = 0;
    // Bayer only: Start_with_green, as Bayer2RGB_/Bayer2Gray_ derive it from the code
    // (demosaicing.cpp:1049-1050, :777-783). It is a field for the same reason `gb` is -- the
    // invoker is handed it already resolved and never sees the code number. Bayer's other phase
    // value, Blue, needs no new field: it is exactly -1 for BG/GB and +1 for RG/GR, which is the
    // `blue` above under a different meaning (0/2 is a channel index, +-1 a polarity), and no
    // op other than Bayer reads that field as a polarity.
    int swg = 0;
    // Xyz/Lab/Luv only: whether the sRGB transfer function applies. The "L" codes of this
    // family (LBGR2Lab, Lab2LBGR, Luv2LBGR, ...) mean LINEAR RGB, i.e. no transfer function at
    // all -- color.hpp is_sRGB() is true for exactly the non-L codes. For Lab this is the whole
    // difference (the CPU's linearGammaTab_b[i] is just i << 3); for Luv it decides which
    // reverse gamma applies, and the forward linear direction is refused entirely.
    bool srgb = true;
    // Xyz/Lab/Luv only: the CPU's nine integer coefficients, in ITS order. They are filled from
    // the code by cvtcolor.cpp's coeffs_for() (which replicates the constructors in color_lab.cpp,
    // including the blueIdx swap and the white-point scaling) rather than written out per row:
    // nine numbers x 18 codes is a table nobody can check by eye, while the constructor is four
    // lines. Zero for every other op.
    int k[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    // Yuv422 only: which byte slot holds chroma U within the 4-byte macropixel -- 0 for UYVY,
    // 1 for YUY2, 2 for YVYU (color_yuv.simd.hpp:1722-1730). Appended at the END of the struct
    // deliberately: inserting it in the middle renumbered every later field and silently moved
    // the Lab/Luv `k[]` initialisers by one position (measured: k[0] read 217 instead of 12615).
    int ord = 0;
};

// The table. A code that is not here is refused by name; there is no nearest-neighbour
// substitution and no fallback to the CPU implementation.
// CV_EXPORTS for the same reason `all()` is: OpenCV builds with hidden visibility and the
// accuracy test checks the coefficient rows against coeffs_for.
CV_EXPORTS const CodeSpec* find(int code);
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
    // Both views have matching depth; normalize owns the family-specific admission.
    bool u16 = false;
    bool f32 = false;
};

// Metadata/admission used by every adapter before output allocation. No device access.
struct Output { const CodeSpec* spec = nullptr; int rows = 0, cols = 0; };
kf_resize_rt::Status resolve_output(const kf_resize_rt::DeviceView& src, int code, Output& out);

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

// The tables XYZ/Lab/Luv read, built EXACTLY as color_lab.cpp's createLabTabs() and
// initLUTforLABLUVs16() build them -- softfloat/softdouble evaluation and cvRound(), not a
// double approximation. That is not pedantry: these tables are cvRound() of a softfloat
// evaluation and the values are then compared for exact equality against a CPU that computed
// them the same way, so a table built with std::pow and lrint differs in a handful of entries
// and every pixel that reads one is off by one. (Anchors that a wrong build gets right anyway,
// which is why they are recorded: sRGBGammaTab_b = 0,1,1,2,2 at 0..4 and 2040 at 255;
// LabCbrtTab_b = 4520,4645,4770,4895 at 0..3 and 37555 at 3071; LabToYF_b at L=0 (0,2260) and
// L=255 (16384,16384).)
//
// The three arrays are kept in the element type the kernel loads them as. `cbrt` is int32 and
// NOT int16 even though the CPU stores ushort: its entries reach 37555, so a signed 16-bit load
// would come back negative. `luv` is int64 for the same reason at the other end -- LvToVpl_b
// reaches 6.2e13 (measured over all 65536 (L,v) pairs), which is why the CPU declares it
// `long long`.
struct LabLuvTables {
    std::vector<int> gamma;       // 256:     sRGBGammaTab_b (or linearGammaTab_b, identical scale)
    std::vector<int> cbrt;        // 3072:    LabCbrtTab_b
    std::vector<int> yf;          // 512:     LabToYF_b, 2 per L
    std::vector<int> abxz;        // 11264:   abToXZ_b, already shifted to start at index 0
    std::vector<int> invgamma;    // 4096:    sRGBInvGammaTab_b
    std::vector<int> lu;          // 65536:   LuToUp_b
    std::vector<int> lv;          // 65536:   LvToVp_b
    std::vector<long long> lvpl;  // 65536:   LvToVpl_b
    std::vector<short> luvLut;    // 33^3*3*8: RGB2LuvLUT_s16, built only for Op::Luv forward
};
// Built on first use and cached for the process: the Luv builder runs 33^3 softfloat
// evaluations plus a 33^3*8 fill, and it does not depend on the code.
const LabLuvTables& labluv_tables();

// The nine integer coefficients of a Xyz/Lab/Luv code, replicated from the CPU's constructors
// in color_lab.cpp (see the definition). Filled into CodeSpec::k when the table is built; zero
// for every other op. The test calls it directly to prove the table's rows and this function
// agree, so a code added to one and not the other cannot pass.
CV_EXPORTS void coeffs_for(int code, int out[9]);

// The offset, in ENTRIES, of each array inside the corresponding device buffer. These are the
// numbers the kernel hardcodes too (kernels/cvtcolor_cpu.py: T_GAMMA ... T_LV), and
// test_cvtcolor.cpp's numeric tests are what would fail first if the two drifted apart.
struct LabLuvLayout {
    static const int gamma = 0;
    static const int cbrt = 256;
    static const int yf = 256 + 3072;
    static const int abxz = 256 + 3072 + 512;
    // abToXZ_b is LAB_BASE*9/4 = 36864 entries, not the 11264 the index RANGE suggests: the CPU
    // allocates for i = minABvalue .. minABvalue + LAB_BASE*9/4 (color_lab.cpp:1062-1063), and
    // the valid indices run to ~23000, so the array cannot be sized from the observed values.
    static const int invgamma = abxz + 36864;
    static const int lu = invgamma + 4096;
    static const int lv = lu + 65536;
    static const int t32_entries = lv + 65536;
};

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
