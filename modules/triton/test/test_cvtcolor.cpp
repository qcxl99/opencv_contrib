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
};

static std::string label(const CvtCase& c, Size sz)
{
    return cv::format("%s %dx%d", c.name, sz.width, sz.height);
}

// Sizes chosen to hit the interesting boundaries of a flat pass over pixels: smaller than one
// BLOCK, exactly one BLOCK, one more, a row that is not a multiple of the block, and a real
// image size. The kernel's masking is the thing under test at the small end.
const Size kSizes[] = {Size(1, 1), Size(13, 7), Size(256, 1), Size(257, 3), Size(97, 61), Size(640, 480)};

TEST(Triton_CvtColor, MatchesCpu)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    uint64 seed = 1;
    cuda::Stream stream;
    for (const CvtCase& c : kCases)
        for (const Size& sz : kSizes)
        {
            SCOPED_TRACE(label(c, sz));
            const Mat host = randomImage(sz, CV_MAKETYPE(CV_8U, c.scn), seed++);
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
        const Size sz(67, 43);
        const Mat host = randomImage(sz, CV_MAKETYPE(CV_8U, c.scn), seed++);
        cuda::GpuMat srcParent, dstParent;
        cuda::GpuMat src = roiOf(srcParent, sz, CV_MAKETYPE(CV_8U, c.scn), Point(3, 1), 7, 3, 0.0);
        cuda::GpuMat dst = roiOf(dstParent, sz, CV_MAKETYPE(CV_8U, c.dcn), Point(5, 2), 9, 5, 77.0);
        src.upload(host);
        const void* before = dst.cudaPtr();
        cv::triton::cvtColor(src, dst, c.code, stream);
        stream.waitForCompletion();
        EXPECT_EQ(before, dst.cudaPtr()) << "the output was reallocated";
        Mat got;
        dst.download(got);
        EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, c.code), NORM_INF));
        EXPECT_TRUE(guardsIntact(dstParent, Rect(Point(5, 2), sz), 77.0)) << "wrote outside the dst ROI";
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
        for (const BoundaryCase& b : boundaryCases())
        {
            SCOPED_TRACE(label(c, sz) + " / " + b.name);
            Mat host(sz, CV_MAKETYPE(CV_8U, c.scn), Scalar::all(0));
            // Written through a byte pointer at the code's channel count. An earlier version
            // wrote `Vec3b px; px[3] = ...` for the 4-channel codes: that is three bytes past the
            // end of px, and `at<Vec3b>` on a 4-channel Mat strides rows by 3*cols instead of
            // 4*cols. Both are undefined behaviour that happened not to crash, so the boundary
            // test could pass without ever placing the intended pixel.
            for (int i = 0; i < (int)b.colours.size(); ++i)
            {
                const Vec3b& v = b.colours[i];
                uchar* px = host.ptr(i / sz.width) + (i % sz.width) * c.scn;
                px[0] = v[0];
                px[1] = v[1];
                px[2] = v[2];
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
                ASSERT_EQ(v[1], px[1]) << "fixture: colour " << i << " byte 1";
                ASSERT_EQ(v[2], px[2]) << "fixture: colour " << i << " byte 2";
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
                SCOPED_TRACE(label(c, sz) + cv::format(" padX=%d", padX));
                const Mat host = randomImage(sz, CV_MAKETYPE(CV_8U, c.scn), seed++);
                cuda::GpuMat srcParent, dstParent;
                cuda::GpuMat src = roiOf(srcParent, sz, CV_MAKETYPE(CV_8U, c.scn),
                                         Point(0, 0), sz.width + padX, 2, 0.0);
                src.upload(host);
                // The destination is an offset ROI inside a padded parent, so both its pitch and
                // its begin pointer differ from src's, and the guard outside it must survive.
                cuda::GpuMat dst = roiOf(dstParent, sz, CV_MAKETYPE(CV_8U, c.dcn),
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
        try { f(); }
        catch (const cv::Exception& e) { return e.code; }
        return 0;
    };
    // The float32 conversions: not served, refused by name rather than under a tolerance.
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(src, dst, COLOR_HSV2BGR); }));
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(src, dst, COLOR_BGR2HLS); }));
    // Outside CV_8U.
    cuda::GpuMat f32(32, 32, CV_32FC3);
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(f32, dst, COLOR_BGR2GRAY); }));
    // A source with the wrong channel count for the code.
    cuda::GpuMat c1(32, 32, CV_8UC1);
    EXPECT_EQ(Error::StsUnsupportedFormat, code([&] { cv::triton::cvtColor(c1, dst, COLOR_BGR2GRAY); }));
    // In-place.
    cuda::GpuMat inplace(32, 32, CV_8UC3);
    EXPECT_EQ(Error::StsBadArg, code([&] { cv::triton::cvtColor(inplace, inplace, COLOR_BGR2RGB); }));
    // A destination that already exists with the wrong channel count is made to fit, not
    // rejected: BGR2BGRA into a 3-channel buffer must produce 4 channels.
    cuda::GpuMat wrong(32, 32, CV_8UC3);
    cv::triton::cvtColor(src, wrong, COLOR_BGR2BGRA);
    cuda::Stream::Null().waitForCompletion();
    EXPECT_EQ(4, wrong.channels());
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
    // The numeric tests do not depend on this one: they compare against the portable C++ oracle
    // in test_oracle.cpp, which is compiled into this binary and cannot be IPP-dispatched. A
    // build that computes different numbers still fails them, which is the check that matters.
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
            "numeric tests use the compiled-in portable oracle and are unaffected");
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

}}  // namespace
