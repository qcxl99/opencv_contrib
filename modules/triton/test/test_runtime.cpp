// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Runtime behaviour behind cv::triton::resize: non-default streams, context isolation,
// table-upload ordering, in-flight resource retention, cache eviction, artifact integrity and the
// condition under which the two scratch arguments may be passed as null.
//
// The ordering and retention tests are deterministic: a host callback blocks one stream, so the
// upload and the launch behind it are known to be pending when the next call is made, and new
// table buffers are poisoned, so reading a table too early produces garbage, not plausible output.

#include "test_helpers.hpp"
#include "../src/runtime/compile_protocol.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>

#include <cuda.h>

namespace opencv_test { namespace {

using namespace triton_test;

static void CUDA_CB hostSleep(void* ms)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(*static_cast<int*>(ms)));
}

static void blockStream(cuda::Stream& s, int* ms)
{
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchHostFunc(cuda::StreamAccessor::getStream(s), hostSleep, ms));
}

static std::string configField(const std::string& key)
{
    const std::string c = td::runtimeConfig();
    const std::size_t a = c.find(key + "=");
    if (a == std::string::npos) return std::string();
    const std::size_t b = c.find(';', a);
    return c.substr(a + key.size() + 1, b == std::string::npos ? std::string::npos : b - a - key.size() - 1);
}

struct Restore
{
    ~Restore() { td::setCacheLimits(64, 256); td::setPoisonNewTables(false); }
};

static void runOne(const Mat& host, Size dsize, int interp, cuda::Stream& stream, cuda::GpuMat& dst)
{
    cuda::GpuMat src(host);
    cv::triton::resize(src, dst, dsize, 0, 0, interp, stream);
    stream.waitForCompletion();
}

// The installed module must find its resources in the install tree, not in a source or build tree.
TEST(Triton_Runtime, ResourcesResolved)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    const std::string res = configField("resources");
    EXPECT_FALSE(res.empty());
    EXPECT_EQ(std::string::npos, td::runtimeConfig().find(";error=")) << td::runtimeConfig();
    const char* want = std::getenv("OPENCV_TRITON_TEST_EXPECT_RESOURCES_PREFIX");
    if (want)
        EXPECT_EQ(0u, res.find(want)) << "resources " << res << " are not under " << want;
    std::cout << "runtime: " << td::runtimeConfig() << std::endl;
}

// Two non-default streams in flight at once, different data, the same key and tables.
TEST(Triton_Runtime, ConcurrentStreams)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    cuda::Stream a(cudaStreamNonBlocking), b(cudaStreamNonBlocking);
    const Size ssz(321, 203), dsz(149, 87);
    for (int round = 0; round < 8; ++round)
    {
        const Mat ha = randomImage(ssz, CV_8UC3, 1000 + round), hb = randomImage(ssz, CV_8UC3, 2000 + round);
        cuda::GpuMat sa(ha), sb(hb), da, db;
        cv::triton::resize(sa, da, dsz, 0, 0, INTER_LINEAR, a);
        cv::triton::resize(sb, db, dsz, 0, 0, INTER_LINEAR, b);
        a.waitForCompletion();
        b.waitForCompletion();
        Mat ga, gb;
        da.download(ga);
        db.download(gb);
        EXPECT_TRUE(meetsContract(ga, oracle(ha, dsz, 0, 0, INTER_LINEAR), INTER_LINEAR));
        EXPECT_TRUE(meetsContract(gb, oracle(hb, dsz, 0, 0, INTER_LINEAR), INTER_LINEAR));
    }
}

// Tables uploaded on stream A, whose upload is still behind a blocked host callback; a call on
// stream B with the same shape must wait for that upload through its event.
TEST(Triton_Runtime, TableUploadOrderingAcrossStreams)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    Restore restore;
    td::setPoisonNewTables(true);
    const Size ssz(211, 97), dsz(131, 61);    // shape unique to this test: its tables are new
    cuda::Stream a(cudaStreamNonBlocking), b(cudaStreamNonBlocking);
    {   // compile/load the binary beforehand, with other tables: the wait below is the upload's
        cuda::GpuMat d;
        runOne(randomImage(Size(53, 41), CV_8UC3, 1), Size(23, 19), INTER_LINEAR, a, d);
    }
    td::reapCompleted();
    td::resetRuntimeStats();
    const Mat ha = randomImage(ssz, CV_8UC3, 11), hb = randomImage(ssz, CV_8UC3, 12);
    cuda::GpuMat sa(ha), sb(hb), da(dsz, CV_8UC3), db(dsz, CV_8UC3);
    int ms = 400;
    const auto t0 = std::chrono::steady_clock::now();
    blockStream(a, &ms);
    cv::triton::resize(sa, da, dsz, 0, 0, INTER_LINEAR, a);
    const double returnedA = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    cv::triton::resize(sb, db, dsz, 0, 0, INTER_LINEAR, b);
    const double returnedB = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    b.waitForCompletion();
    const double waited = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    a.waitForCompletion();
    const td::RuntimeStats st = td::runtimeStats();
    EXPECT_EQ(1, st.table_uploads);
    EXPECT_EQ(1, st.table_hits);
    EXPECT_EQ(1, st.cross_stream_waits);
    // Neither call stalled the host (the module was loaded beforehand), so B's wait is the event wait.
    EXPECT_LT(returnedA, 0.5 * ms) << "the first call stalled the host: the scenario tests nothing";
    EXPECT_LT(returnedB, 0.5 * ms) << "the second call stalled the host: the scenario tests nothing";
    EXPECT_GE(waited, 0.8 * ms) << "stream B did not wait for the upload behind stream A's block";
    Mat ga, gb;
    da.download(ga);
    db.download(gb);
    EXPECT_TRUE(meetsContract(ga, oracle(ha, dsz, 0, 0, INTER_LINEAR), INTER_LINEAR));
    EXPECT_TRUE(meetsContract(gb, oracle(hb, dsz, 0, 0, INTER_LINEAR), INTER_LINEAR));
}

// A launch still in flight holds its module and tables: evicting them from the caches frees nothing
// until the launch's completion has been observed. Module load and unload block until the context
// drains, so eviction under a truly pending launch must not need a load: both kernels are cached
// first; the tables are evicted by another shape of the same kernel, the module by shrinking.
TEST(Triton_Runtime, InFlightRetentionAndEviction)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    Restore restore;
    const Size sa(141, 83), da(67, 59), sb(123, 77), db(71, 43), sc(119, 73), dc(61, 47);   // unique to this test
    cuda::Stream a(cudaStreamNonBlocking), b(cudaStreamNonBlocking);
    {   // both binaries are on disk and loaded
        cuda::GpuMat d1, d2;
        runOne(randomImage(Size(31, 29), CV_8UC1, 3), Size(13, 11), INTER_LINEAR, a, d2);
        runOne(randomImage(Size(31, 29), CV_8UC1, 2), Size(13, 11), INTER_NEAREST, a, d1);
    }
    td::setCacheLimits(2, 1);
    td::setPoisonNewTables(true);
    td::reapCompleted();
    td::resetRuntimeStats();

    const Mat ha = randomImage(sa, CV_8UC1, 21), hb = randomImage(sb, CV_8UC1, 22), hc = randomImage(sc, CV_8UC1, 23);
    cuda::GpuMat ga(ha), gb(hb), gc(hc), oa(da, CV_8UC1), ob(db, CV_8UC1), oc(dc, CV_8UC1);
    int ms = 800;
    const auto t0 = std::chrono::steady_clock::now();
    blockStream(a, &ms);
    cv::triton::resize(ga, oa, da, 0, 0, INTER_NEAREST, a);   // upload and launch sit behind the block
    cv::triton::resize(gb, ob, db, 0, 0, INTER_NEAREST, b);   // same kernel, other tables: evicts A's tables
    b.waitForCompletion();                                    // B runs to completion while A is blocked
    cv::triton::resize(gc, oc, dc, 0, 0, INTER_LINEAR, b);    // the linear module becomes most recent
    td::setCacheLimits(1, 1);                                 // evicts the nearest module A is using
    td::reapCompleted();
    td::RuntimeStats st = td::runtimeStats();
    const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_LT(elapsed, 0.5 * ms) << "A's block had nearly ended: the scenario tests nothing";
    EXPECT_EQ(0, st.module_loads) << "a load would have drained the context";
    EXPECT_GE(st.table_evictions, 1);
    EXPECT_GE(st.module_evictions, 1);
    EXPECT_GE(st.deferred, 2) << "the evicted module and tables should still be held by the blocked launch";
    EXPECT_EQ(0, st.module_frees) << "a module in use was unloaded";
    EXPECT_GE(st.pending, 1);
    // Exactly the table A holds is evicted but not freed: A's new table was evicted by B, while the
    // warm-up's table (evicted by A) and B's table (evicted by C, B has completed) are gone.
    EXPECT_EQ(st.table_evictions - 1, st.table_frees) << "tables in use were freed, or finished ones kept";
    const long long tableFreesWhileBlocked = st.table_frees;

    a.waitForCompletion();
    b.waitForCompletion();
    td::reapCompleted();
    st = td::runtimeStats();
    EXPECT_GE(st.module_frees, 1) << "the evicted module was not released after its launch completed";
    EXPECT_GT(st.table_frees, tableFreesWhileBlocked) << "A's tables were not released after its launch completed";
    EXPECT_EQ(st.table_evictions, st.table_frees);
    EXPECT_EQ(0, st.pending);
    Mat ra, rb, rc;
    oa.download(ra);
    ob.download(rb);
    oc.download(rc);
    EXPECT_TRUE(meetsContract(ra, oracle(ha, da, 0, 0, INTER_NEAREST), INTER_NEAREST));
    EXPECT_TRUE(meetsContract(rb, oracle(hb, db, 0, 0, INTER_NEAREST), INTER_NEAREST));
    EXPECT_TRUE(meetsContract(rc, oracle(hc, dc, 0, 0, INTER_LINEAR), INTER_LINEAR));

    // Evicted is not lost: the same call reloads from disk, re-uploads, compiles nothing.
    const long long compiles = st.helper_compiles, loads = st.module_loads, uploads = st.table_uploads, disk = st.disk_hits;
    cuda::GpuMat oa2;
    runOne(ha, da, INTER_NEAREST, a, oa2);
    st = td::runtimeStats();
    EXPECT_EQ(compiles, st.helper_compiles);
    EXPECT_EQ(loads + 1, st.module_loads);
    EXPECT_EQ(disk + 1, st.disk_hits);
    EXPECT_EQ(uploads + 1, st.table_uploads);
    oa2.download(ra);
    EXPECT_TRUE(meetsContract(ra, oracle(ha, da, 0, 0, INTER_NEAREST), INTER_NEAREST));
}

// Modules and tables belong to one context: another context on the same device loads and
// uploads its own, and a destroyed context's resources are never reused, even if its handle
// value comes back.
TEST(Triton_Runtime, ContextIsolation)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    Restore restore;
    const Size ssz(90, 70), dsz(47, 35);
    const Mat host = randomImage(ssz, CV_8UC3, 31);
    const Mat want = oracle(host, dsz, 0, 0, INTER_LINEAR);
    {   // the primary context has this key and these tables
        cuda::GpuMat d;
        cuda::Stream s;
        runOne(host, dsz, INTER_LINEAR, s, d);
    }
    CUdevice dev;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&dev, cv::cuda::getDevice()));
    CUcontext primary = nullptr;
    ASSERT_EQ(CUDA_SUCCESS, cuCtxGetCurrent(&primary));

    for (int round = 0; round < 2; ++round)
    {
        SCOPED_TRACE(cv::format("private context %d", round));
        CUcontext ctx = nullptr;
        ASSERT_EQ(CUDA_SUCCESS, cuCtxCreate(&ctx, 0, dev));     // now current
        const std::size_t sstep = ssz.width * 3, dstep = dsz.width * 3;
        CUdeviceptr ps = 0, pd = 0;
        ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&ps, sstep * ssz.height));
        ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&pd, dstep * dsz.height));
        Mat dense = host.clone();
        ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD(ps, dense.data, sstep * ssz.height));
        td::reapCompleted();
        td::resetRuntimeStats();
        {
            cuda::GpuMat gs(ssz.height, ssz.width, CV_8UC3, reinterpret_cast<void*>(ps), sstep);
            cuda::GpuMat gd(dsz.height, dsz.width, CV_8UC3, reinterpret_cast<void*>(pd), dstep);
            cv::triton::resize(gs, gd, dsz, 0, 0, INTER_LINEAR);
            ASSERT_EQ(CUDA_SUCCESS, cuCtxSynchronize());
        }
        const td::RuntimeStats st = td::runtimeStats();
        EXPECT_EQ(1, st.module_loads) << "the module of another context was reused";
        EXPECT_EQ(1, st.table_uploads) << "the tables of another context were reused";
        EXPECT_EQ(0, st.helper_compiles) << "the binary is the same; only its loading is per context";
        Mat got(dsz, CV_8UC3);
        ASSERT_EQ(CUDA_SUCCESS, cuMemcpyDtoH(got.data, pd, dstep * dsz.height));
        EXPECT_TRUE(meetsContract(got, want, INTER_LINEAR));
        td::reapCompleted();
        EXPECT_EQ(0, td::runtimeStats().pending);
        cuMemFree(ps);
        cuMemFree(pd);
        ASSERT_EQ(CUDA_SUCCESS, cuCtxDestroy(ctx));             // its cache entries are now stale
        ASSERT_EQ(CUDA_SUCCESS, cuCtxSetCurrent(primary));
    }
    // Stale entries are dropped without touching the destroyed context, and the primary still works.
    td::setCacheLimits(1, 1);
    td::setCacheLimits(64, 256);
    cuda::GpuMat d;
    cuda::Stream s;
    runOne(host, dsz, INTER_LINEAR, s, d);
    Mat got;
    d.download(got);
    EXPECT_TRUE(meetsContract(got, want, INTER_LINEAR));
}

TEST(Triton_Runtime, SecondDevice)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    if (cv::cuda::getCudaEnabledDeviceCount() < 2) throw SkipTestException("one visible device");
    const int home = cv::cuda::getDevice();
    const Mat host = randomImage(Size(77, 55), CV_32FC3, 41);
    const Mat want = oracle(host, Size(39, 28), 0, 0, INTER_LINEAR);
    td::reapCompleted();
    td::resetRuntimeStats();
    cv::cuda::setDevice(home == 0 ? 1 : 0);
    {
        cuda::GpuMat d;
        cuda::Stream s;
        runOne(host, Size(39, 28), INTER_LINEAR, s, d);
        Mat got;
        d.download(got);
        EXPECT_TRUE(meetsContract(got, want, INTER_LINEAR));
    }
    EXPECT_GE(td::runtimeStats().module_loads, 1);
    td::reapCompleted();
    cv::cuda::setDevice(home);
}

// What the launcher runs is what the manifest says: every loaded artifact verifies independently
// of the runtime, and a corrupted one on disk is rejected and rebuilt, not loaded.
TEST(Triton_Runtime, ArtifactIntegrity)
{
    if (!ready()) throw SkipTestException("no Triton backend");
    Restore restore;
    const Mat host = randomImage(Size(93, 67), CV_32FC1, 51);
    const Size dsz(41, 37);
    cuda::Stream s;
    cuda::GpuMat d;
    runOne(host, dsz, INTER_NEAREST, s, d);
    td::RuntimeStats st = td::runtimeStats();
    const std::string dir = st.last_artifact, key = st.last_key;
    ASSERT_FALSE(dir.empty());
    kf_resize_rt::Manifest m;
    std::string why;
    ASSERT_TRUE(kf_resize_rt::verify_artifact(dir, key, m, why)) << why;
    EXPECT_TRUE(kf_resize_rt::launch_abi_ok(m, why)) << why;
    EXPECT_EQ("_resize_cpu_nearest_kernel", m.function);
    EXPECT_EQ(0, m.global_scratch_bytes);
    EXPECT_EQ(0, m.profile_scratch_bytes);

    // Corrupt it on disk, push it out of the in-process cache, ask again.
    td::setCacheLimits(1, 1);
    {
        cuda::GpuMat other;
        runOne(randomImage(Size(31, 29), CV_32FC1, 52), Size(13, 11), INTER_LINEAR, s, other);
    }
    td::reapCompleted();
    {
        std::fstream f(dir + "/" + m.binary_file, std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(f.good());
        f.seekg(static_cast<std::streamoff>(m.binary_bytes / 2));
        char c = 0;
        f.read(&c, 1);
        c = static_cast<char>(c ^ 0xFF);
        f.seekp(static_cast<std::streamoff>(m.binary_bytes / 2));
        f.write(&c, 1);
    }
    kf_resize_rt::Manifest bad;
    EXPECT_FALSE(kf_resize_rt::verify_artifact(dir, key, bad, why));
    td::resetRuntimeStats();
    cuda::GpuMat d2;
    runOne(host, dsz, INTER_NEAREST, s, d2);
    st = td::runtimeStats();
    EXPECT_EQ(1, st.disk_rejected);
    EXPECT_EQ(1, st.helper_compiles);
    kf_resize_rt::Manifest again;
    EXPECT_TRUE(kf_resize_rt::verify_artifact(dir, key, again, why)) << why;
    Mat got;
    d2.download(got);
    EXPECT_TRUE(meetsContract(got, oracle(host, dsz, 0, 0, INTER_NEAREST), INTER_NEAREST));
}

// The two scratch arguments are null only under a validated compiler ABI and zero scratch bytes.
TEST(Triton_Runtime, ScratchArgumentsAreConditional)
{
    using kf_resize_rt::Manifest;
    Manifest ok;
    ok.compiler = "triton-3.6.0";
    ok.global_scratch_bytes = 0;
    ok.profile_scratch_bytes = 0;
    ok.abi = {{"src", "*u8", "u64", false}, {"W", "i32", "u32", false},
              {"global_scratch", "*i8", "u64", true}, {"profile_scratch", "*i8", "u64", true}};
    std::string why;
    EXPECT_TRUE(kf_resize_rt::launch_abi_ok(ok, why)) << why;
    Manifest m = ok;
    m.compiler = "triton-9.9.9";
    EXPECT_FALSE(kf_resize_rt::launch_abi_ok(m, why)) << "a compiler whose ABI was not validated";
    m = ok;
    m.global_scratch_bytes = 1024;
    EXPECT_FALSE(kf_resize_rt::launch_abi_ok(m, why)) << "the kernel needs scratch memory";
    m = ok;
    m.profile_scratch_bytes = -1;
    EXPECT_FALSE(kf_resize_rt::launch_abi_ok(m, why)) << "scratch size unknown";
    m = ok;
    m.abi.pop_back();
    EXPECT_FALSE(kf_resize_rt::launch_abi_ok(m, why)) << "only one trailing pointer";
    m = ok;
    m.abi[2].name = "something_else";
    EXPECT_FALSE(kf_resize_rt::launch_abi_ok(m, why)) << "not the expected trailing pointers";
}

}}  // namespace
