// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright 2026, The FlagOS Contributors.
// Shared accuracy through the public backend entries, without SDK-specific headers.
// Device fixtures exercise pitched/ROI images; references check sampling and rounding.
// Guard tests detect writes outside the destination view, including 16U byte damage.
#include "../validation/device.hpp"
#include "opencv2/triton/device.hpp"
#include "../validation/guards.hpp"
#include <limits>
#include <cstring>
#include <cstdint>
#include <cfenv>
#include "../validation/reference.hpp"
#include "../validation/resize_cases.hpp"
#include "../validation/cvtcolor_cases.hpp"
#include "../src/runtime/provider.hpp"
namespace opencv_test
{
namespace
{
namespace device = triton_validation;
using namespace triton_test;

static void requireDevice()
{
    if (!device::available())
        throw SkipTestException("no usable validation backend");
}

// CPU-only control: supported depths/channels, separate row pitches, and byte corruption.
TEST(Triton_Validation, GuardBandsBytewise)
{
    for (int depth : {CV_8U, CV_16U, CV_32F})
        for (int channels : {1, 3, 4})
        {
            const int type = CV_MAKETYPE(depth, channels);
            SCOPED_TRACE(cv::format("depth=%d channels=%d", depth, channels));
            Mat beforeStorage(7, 12, type, Scalar::all(77));
            Mat afterStorage(7, 15, type, Scalar::all(77));
            const Mat before = beforeStorage(Rect(1, 1, 9, 5));
            Mat after = afterStorage(Rect(2, 1, 9, 5));
            const Rect changed(2, 1, 4, 3);
            EXPECT_TRUE(guardsEqual(before, after, changed));
            after(changed).setTo(Scalar::all(13));
            EXPECT_TRUE(guardsEqual(before, after, changed));
            after.ptr(0)[after.elemSize() - 1] ^= 1;
            EXPECT_FALSE(guardsEqual(before, after, changed));
            EXPECT_FALSE(guardsEqual(before, after(Rect(0, 0, 8, 5)), changed));
            const Mat otherType(after.size(), CV_MAKETYPE(depth, channels == 1 ? 3 : 1));
            EXPECT_FALSE(guardsEqual(before, otherType, changed));
        }
}

TEST(Triton_Common, OutputIndexAdmissionNeedsNoAllocation)
{
    using namespace cv_triton_runtime;
    const int limit = std::numeric_limits<int>::max();
    EXPECT_TRUE(validate_output_shape(1, limit, 1).ok());
    EXPECT_TRUE(validate_output_shape(1, limit / 4, 4).ok());
    EXPECT_EQ(Reason::IndexRange, validate_output_shape(2, limit / 4, 4).reason);
    EXPECT_EQ(Reason::IndexRange, validate_output_shape(limit, limit, 4).reason);
    EXPECT_EQ(Reason::EmptyOrSize, validate_output_shape(0, 1, 1).reason);
    ResizeRequest request;
    request.src.rows = request.src.cols = 1;
    request.src.channels = 3;
    request.dsize_h = 2;
    request.dsize_w = limit / 3;
    cv_triton_resize::Size2 resolved;
    EXPECT_EQ(Reason::IndexRange, resolve_output(request, resolved).reason);
}

TEST(Triton_Common, RowOverlapExcludesPadding)
{
    using namespace cv_triton_runtime;
    DeviceView first;
    first.ptr = 4096;
    first.rows = 5;
    first.cols = 8;
    first.channels = 3;
    first.step = 128;
    DeviceView second = first;
    second.ptr += first.row_bytes();
    ASSERT_TRUE(validate_view(first, "first").ok());
    ASSERT_TRUE(validate_view(second, "second").ok());
    EXPECT_FALSE(views_overlap(first, second));
    EXPECT_FALSE(views_overlap(second, first));
    --second.ptr;
    EXPECT_TRUE(views_overlap(first, second));
    EXPECT_TRUE(views_overlap(second, first));
    EXPECT_TRUE(views_overlap(first, first));
    second = first;
    second.ptr += first.step * first.rows;
    EXPECT_FALSE(views_overlap(first, second));
}

TEST(Triton_Common, DisjointHorizontalRoisShareAllocation)
{
    requireDevice();
    device::Stream stream;
    const Size size(17, 13);
    const Rect sourceRect(1, 2, size.width, size.height);
    const Rect destinationRect(23, 2, size.width, size.height);
    const Mat matrix = Mat::eye(2, 3, CV_64F);
    for (int operation = 0; operation < 3; ++operation)
    {
        SCOPED_TRACE(operation);
        Mat parent(19, 47, CV_8UC3, Scalar::all(37));
        const Mat source = randomImage(size, CV_8UC3, 78001 + operation);
        source.copyTo(parent(sourceRect));
        device::Image storage;
        storage.upload(parent);
        device::Image src(storage, sourceRect), dst(storage, destinationRect);
        const auto address = dst.address();
        Mat reference;
        if (operation == 0)
        {
            reference = oracle(source, size, 0, 0, INTER_NEAREST);
            device::resize(src, dst, size, 0, 0, INTER_NEAREST, stream);
        }
        else if (operation == 1)
        {
            reference = cpuCvt(source, COLOR_BGR2RGB);
            device::cvtColor(src, dst, COLOR_BGR2RGB, stream);
        }
        else
        {
            reference =
                warpReference(source, matrix, size, INTER_NEAREST, BORDER_REPLICATE, Scalar());
            device::warpAffine(src, dst, matrix, size, INTER_NEAREST, BORDER_REPLICATE, Scalar(),
                               stream);
        }
        stream.wait();
        EXPECT_EQ(address, dst.address());
        EXPECT_EQ(0, cv::norm(reference, dst.download(), NORM_INF));
        EXPECT_TRUE(guardsEqual(parent, storage.download(), destinationRect));
    }
}

TEST(Triton_Common, ResizeRoutesLayoutsAndStreams)
{
    requireDevice();
    uint64 seed = 73001;
    for (bool useDefault : {false, true})
    {
        device::Stream stream(useDefault);
        for (const auto &c : resize_cases::kCases)
            for (int type : resize_cases::kTypes)
            {
                SCOPED_TRACE(cv::format("backend=%s mode=%d type=%d default=%d", device::backend(),
                                        c.interp, type, useDefault));
                const Mat host = randomImage(c.src, type, seed++);
                const Mat reference = oracle(host, c.dst, c.fx, c.fy, c.interp);
                Mat sourceParent(c.src + Size(13, 7), type, Scalar::all(23));
                host.copyTo(sourceParent(Rect(Point(1, 2), c.src)));
                device::Image sourceStorage;
                sourceStorage.upload(sourceParent);
                device::Image src(sourceStorage, Rect(Point(1, 2), c.src));
                const Mat initial(reference.size() + Size(17, 9), type, Scalar::all(37));
                device::Image targetStorage;
                targetStorage.upload(initial);
                const Rect targetRect(Point(3, 4), reference.size());
                device::Image dst(targetStorage, targetRect);
                const auto address = dst.address();
                device::resize(src, dst, c.dst, c.fx, c.fy, c.interp, stream);
                stream.wait();
                EXPECT_EQ(address, dst.address());
                EXPECT_TRUE(meetsContract(dst.download(), reference, c.interp));
                EXPECT_TRUE(guardsEqual(initial, targetStorage.download(), targetRect));
            }
    }
}

static void checkCvt(const cvt_cases::CvtCase &c, uint64 seed, device::Stream &stream)
{
    const Size size(34, c.yuv420 ? 24 : 17);
    const int type = CV_MAKETYPE(c.depth, c.scn);
    Mat host = randomImage(size, type, seed);
    if (c.depth == CV_32F)
        for (int y = 0; y < host.rows; ++y)
            for (int x = 0; x < host.cols; ++x)
                if ((x + y) % 3 == 0)
                    for (int channel = 0; channel < c.scn; ++channel)
                        host.ptr<float>(y)[x * c.scn + channel] *= 32767.5f;
    const Mat reference = cpuCvt(host, c.code);
    Mat parent(size + Size(12, 7), type, Scalar::all(17));
    host.copyTo(parent(Rect(Point(2, 1), size)));
    device::Image storage;
    storage.upload(parent);
    device::Image src(storage, Rect(Point(2, 1), size));
    const Mat initial(reference.size() + Size(13, 8), reference.type(), Scalar::all(43));
    device::Image target;
    target.upload(initial);
    const Rect outputRect(Point(1, 3), reference.size());
    device::Image dst(target, outputRect);
    const auto address = dst.address();
    device::cvtColor(src, dst, c.code, stream);
    stream.wait();
    ASSERT_EQ(address, dst.address());
    const Mat got = dst.download();
    ASSERT_EQ(reference.size(), got.size());
    ASSERT_EQ(reference.type(), got.type());
    EXPECT_TRUE(meetsCvtContract(got, reference, host, cvt_cases::floatArithmetic(c.code),
                                 cvt_cases::floatArithmetic(c.code) &&
                                     cv_triton_color::float_cpu_separate4()));
    EXPECT_TRUE(guardsEqual(initial, target.download(), outputRect));
}

TEST(Triton_Common, CvtColorCriterionUsesLocalPixelScale)
{
    Mat input(1, 2, CV_32FC3, Scalar::all(0));
    input.at<Vec3f>(0, 1) = Vec3f(65535.f, 1.f, -65535.f);
    Mat reference(1, 2, CV_32FC1, Scalar::all(0));
    Mat got = reference.clone();
    got.at<float>(0, 1) = 0.02f;
    EXPECT_TRUE(meetsCvtContract(got, reference, input, true));
    EXPECT_FALSE(meetsCvtContract(got, reference, input, true, true));
    // The neighbouring large pixel must not lend its tolerance to this one.
    got.at<float>(0, 0) = 0.001f;
    EXPECT_FALSE(meetsCvtContract(got, reference, input, true));
    EXPECT_FALSE(meetsCvtContract(got, reference, input, false));
    got = reference.clone();
    got.at<float>(0, 0) = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(meetsCvtContract(got, reference, input, true));
    Mat exact(1, 2, CV_16UC1, Scalar::all(65535));
    Mat changed = exact.clone();
    changed.at<ushort>(0, 0) = 65534;
    EXPECT_FALSE(meetsCvtContract(changed, exact, exact, false));
}

// Host controls distinguish the original scaled criterion from the conditional cap.
TEST(Triton_Validation, CvtColorConditionalAbsoluteCap)
{
    const double scale = 65535.;
    EXPECT_GT(triton_validation::numeric::cvtF32Tolerance(0., scale), .015625);
    EXPECT_EQ(1.e-4, triton_validation::numeric::cvtF32Tolerance(0., scale, true));
    const int oldRound = std::fegetround();
    if (std::fesetround(FE_DOWNWARD) == 0)
    {
        EXPECT_FALSE(cv_triton_color::float_cpu_separate4());
        std::fesetround(oldRound);
    }
}

TEST(Triton_Common, CvtColorMixedScaleAbsoluteCap)
{
    requireDevice();
    struct Restore
    {
        bool value = cv::useOptimized();
        ~Restore()
        {
            cv::setUseOptimized(value);
        }
    } restore;
    long long elements = 0;
    double worst = 0.;
    for (bool optimized : {false, true})
    {
        cv::setUseOptimized(optimized);
        if (!cv_triton_color::float_cpu_separate4())
            throw SkipTestException(
                "linked CPU arithmetic is not identified separate4; absolute accuracy check unavailable");
        device::Stream stream;
        for (const auto &c : cvt_cases::kCases32F)
            for (int width : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 19, 31, 32, 33, 63, 65})
                for (bool roi : {false, true})
                    for (float scale : {1.f, 255.f, 65535.f})
                    {
                        SCOPED_TRACE(cv::format("%s width=%d roi=%d optimized=%d scale=%g", c.name,
                                                width, roi, optimized, scale));
                        Mat parent(7, width + 7, CV_MAKETYPE(CV_32F, c.scn));
                        RNG rng(12345 + width);
                        rng.fill(parent, RNG::UNIFORM, -2., 2.);
                        Mat host = parent(Rect(2, 1, width, 5));
                        if (!roi)
                            host = host.clone();
                        for (int y = 0; y < host.rows; ++y)
                            for (int x = 0; x < width; ++x)
                                for (int k = 0; k < c.scn; ++k)
                                    host.ptr<float>(y)[x * c.scn + k] *= (x + y) % 3 ? scale : 1.f;
                        device::Image sourceParent;
                        sourceParent.upload(roi ? parent : host);
                        device::Image source =
                            roi ? device::Image(sourceParent, Rect(2, 1, width, 5)) : sourceParent;
                        const Mat initial(9, width + 9, CV_MAKETYPE(CV_32F, c.dcn),
                                          Scalar::all(17));
                        device::Image target;
                        target.upload(initial);
                        const Rect region(2, 2, width, 5);
                        device::Image output = roi ? device::Image(target, region)
                                                   : device::Image(Size(width, 5), initial.type());
                        const auto address = output.address();
                        device::cvtColor(source, output, c.code, stream);
                        stream.wait();
                        const Mat got = output.download(), ref = cpuCvt(host, c.code);
                        ASSERT_EQ(address, output.address());
                        ASSERT_TRUE(meetsCvtContract(got, ref, host,
                                                     cvt_cases::floatArithmetic(c.code), true));
                        const double err = cv::norm(got, ref, NORM_INF);
                        ASSERT_LE(err, 1.e-4);
                        worst = std::max(worst, err);
                        elements += got.total() * got.channels();
                        if (roi)
                        {
                            EXPECT_TRUE(guardsEqual(initial, target.download(), region));
                        }
                    }
    }
    std::cout << "cvtColor strict mixed-scale elements=" << elements << " maxabs=" << worst
              << std::endl;
}

TEST(Triton_Common, CvtColorAllDeclaredCodesAndDepths)
{
    requireDevice();
    uint64 seed = 74001;
    for (bool useDefault : {false, true})
    {
        device::Stream stream(useDefault);
        for (const auto &c : cvt_cases::kCases)
        {
            SCOPED_TRACE(c.name);
            checkCvt(c, seed++, stream);
        }
        for (const auto &c : cvt_cases::kCases16U)
        {
            SCOPED_TRACE(c.name);
            checkCvt(c, seed++, stream);
        }
        for (const auto &c : cvt_cases::kCases32F)
        {
            SCOPED_TRACE(c.name);
            checkCvt(c, seed++, stream);
        }
    }
}

// Word-wide byte paths must preserve tails, row offsets and the generic fallback.
TEST(Triton_Common, PackedByteRoutesAndFallbacks)
{
    requireDevice();
    device::Stream stream;
    const Mat matrix = (Mat_<double>(2, 3) << .9, -.43589, -23.5, .43589, .9, 14.5);
    uint64 seed = 75101;
    for (int width : {255, 256, 257})
        for (int offset : {1, 4})
        {
            const Size size(width, 256);
            for (int code : {COLOR_BGR2BGRA, COLOR_BGRA2BGR, COLOR_BGR2RGB, COLOR_BGRA2GRAY})
            {
                const int scn = code == COLOR_BGRA2BGR || code == COLOR_BGRA2GRAY ? 4 : 3;
                const Mat host = randomImage(size, CV_MAKETYPE(CV_8U, scn), seed++);
                const Mat ref = cpuCvt(host, code);
                Mat parent(size + Size(8, 2), host.type(), Scalar::all(17));
                host.copyTo(parent(Rect(Point(offset, 1), size)));
                device::Image storage; storage.upload(parent);
                device::Image src(storage, Rect(Point(offset, 1), size));
                const Mat initial(size + Size(8, 2), ref.type(), Scalar::all(37));
                device::Image target; target.upload(initial);
                const Rect region(Point(offset, 1), size);
                device::Image dst(target, region);
                device::cvtColor(src, dst, code, stream);
                stream.wait();
                EXPECT_EQ(0, cv::norm(ref, dst.download(), NORM_INF));
                EXPECT_TRUE(guardsEqual(initial, target.download(), region));
            }
            for (int mode : {INTER_NEAREST, INTER_LINEAR})
            {
                const Mat host = randomImage(size, CV_8UC4, seed++);
                const Scalar value(17, 23, 49, 91);
                const Mat ref = warpReference(host, matrix, size, mode | WARP_INVERSE_MAP,
                                               BORDER_REPLICATE, value);
                device::Image src; src.upload(host);
                const Mat initial(size + Size(8, 2), CV_8UC4, Scalar::all(37));
                device::Image target; target.upload(initial);
                const Rect region(Point(offset, 1), size);
                device::Image dst(target, region);
                device::warpAffine(src, dst, matrix, size, mode | WARP_INVERSE_MAP,
                                   BORDER_REPLICATE, value, stream);
                stream.wait();
                EXPECT_EQ(0, cv::norm(ref, dst.download(), NORM_INF));
                EXPECT_TRUE(guardsEqual(initial, target.download(), region));
            }
        }
}

TEST(Triton_Common, ChannelInterpolationBoundaries)
{
    requireDevice();
    device::Stream stream;
    uint64 seed = 75201;
    for (const Size size : {Size(255, 257), Size(256, 256), Size(257, 257)})
        for (int depth : {CV_8U, CV_16U, CV_32F})
            for (int channels : {1, 3, 4})
                for (int mode : {INTER_LINEAR, INTER_CUBIC})
                    for (double slope : {0.0, 0.1249, 0.125, 0.1251, 0.43589})
                    {
                        SCOPED_TRACE(cv::format("width=%d depth=%d cn=%d mode=%d slope=%.5f",
                                                size.width, depth, channels, mode, slope));
                        const int type = CV_MAKETYPE(depth, channels);
                        const Mat host = randomImage(size, type, seed++);
                        Mat parent(size + Size(8, 2), type, Scalar::all(17));
                        const Rect sourceRegion(Point(1, 1), size);
                        host.copyTo(parent(sourceRegion));
                        device::Image sourceStorage;
                        sourceStorage.upload(parent);
                        device::Image src(sourceStorage, sourceRegion);
                        const Mat initial(size + Size(8, 2), type, Scalar::all(37));
                        device::Image target;
                        target.upload(initial);
                        const Rect region(Point(1, 1), size);
                        device::Image dst(target, region);
                        const auto address = dst.address();
                        const Mat matrix = (Mat_<double>(2, 3) << .9, -.43589, -23.5,
                                            slope, .9, 14.5);
                        const Mat ref = warpReference(host, matrix, size, mode | WARP_INVERSE_MAP,
                                                       BORDER_REPLICATE, Scalar());
                        device::warpAffine(src, dst, matrix, size, mode | WARP_INVERSE_MAP,
                                           BORDER_REPLICATE, Scalar(), stream);
                        stream.wait();
                        const Mat got = dst.download();
                        ASSERT_EQ(address, dst.address());
                        if (depth != CV_32F)
                            EXPECT_LE(cv::norm(ref, got, NORM_INF), depth == CV_8U ? 0 : 1);
                        else
                            for (int y = 0; y < got.rows; ++y)
                                for (int x = 0; x < got.cols * channels; ++x)
                                    EXPECT_LE(std::abs(double(ref.ptr<float>(y)[x]) -
                                                       got.ptr<float>(y)[x]),
                                              warpF32Tolerance(ref.ptr<float>(y)[x]));
                        EXPECT_TRUE(guardsEqual(initial, target.download(), region));
                    }
}

TEST(Triton_Common, ChannelModuloBorders)
{
    requireDevice();
    device::Stream stream;
    uint64 seed = 75201;
    for (const Size size : {Size(263, 257)})
        for (int depth : {CV_8U, CV_16U, CV_32F})
            for (int channels : {3, 4})
                for (int mode : {INTER_LINEAR, INTER_CUBIC})
                    for (int border : {BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101})
                    {
                        SCOPED_TRACE(cv::format("width=%d depth=%d cn=%d mode=%d border=%d",
                                                size.width, depth, channels, mode, border));
                        const int type = CV_MAKETYPE(depth, channels);
                        const Mat host = randomImage(size, type, seed++);
                        Mat parent(size + Size(8, 2), type, Scalar::all(17));
                        const Rect sourceRegion(Point(1, 1), size);
                        host.copyTo(parent(sourceRegion));
                        device::Image sourceStorage;
                        sourceStorage.upload(parent);
                        device::Image src(sourceStorage, sourceRegion);
                        const Mat initial(size + Size(8, 2), type, Scalar::all(37));
                        device::Image target;
                        target.upload(initial);
                        const Rect region(Point(1, 1), size);
                        device::Image dst(target, region);
                        const auto address = dst.address();
                        const Mat matrix = (Mat_<double>(2, 3) << .9, -.43589, -23.5,
                                            .43589, .9, 14.5);
                        const Mat ref = warpReference(host, matrix, size, mode | WARP_INVERSE_MAP,
                                                       border, Scalar());
                        device::warpAffine(src, dst, matrix, size, mode | WARP_INVERSE_MAP,
                                           border, Scalar(), stream);
                        stream.wait();
                        const Mat got = dst.download();
                        ASSERT_EQ(address, dst.address());
                        if (depth != CV_32F)
                            EXPECT_LE(cv::norm(ref, got, NORM_INF), depth == CV_8U ? 0 : 1);
                        else
                            for (int y = 0; y < got.rows; ++y)
                                for (int x = 0; x < got.cols * channels; ++x)
                                    EXPECT_LE(std::abs(double(ref.ptr<float>(y)[x]) -
                                                       got.ptr<float>(y)[x]),
                                              warpF32Tolerance(ref.ptr<float>(y)[x]));
                        EXPECT_TRUE(guardsEqual(initial, target.download(), region));
                    }
}

TEST(Triton_Common, ChannelBlendBorders)
{
    requireDevice();
    device::Stream stream;
    uint64 seed = 75301;
    const Size size(263, 257);
    for (int depth : {CV_8U, CV_16U, CV_32F})
        for (int channels : {3, 4})
            for (int border : {BORDER_CONSTANT, BORDER_TRANSPARENT})
                for (int mode : {INTER_LINEAR, INTER_CUBIC})
                {
                    if (border == BORDER_TRANSPARENT && mode == INTER_CUBIC)
                        continue; // Explicitly unsupported by the public contract.
                    for (bool outside : {false, true})
                    {
                        SCOPED_TRACE(cv::format("depth=%d cn=%d mode=%d border=%d outside=%d",
                                                depth, channels, mode, border, outside));
                        const int type = CV_MAKETYPE(depth, channels);
                        const Mat host = randomImage(size, type, seed++);
                        Mat parent(size + Size(8, 2), type, Scalar::all(17));
                        const Rect sourceRegion(Point(1, 1), size);
                        host.copyTo(parent(sourceRegion));
                        device::Image sourceStorage;
                        sourceStorage.upload(parent);
                        device::Image src(sourceStorage, sourceRegion);
                        const Mat initial(size + Size(8, 2), type, Scalar::all(37));
                        device::Image target;
                        target.upload(initial);
                        const Rect region(Point(1, 1), size);
                        device::Image dst(target, region);
                        const Scalar value(0.5, 11, 23, 47);
                        const Mat matrix = (Mat_<double>(2, 3) << .9, -.43589,
                                            outside ? -2048.5 : -23.5,
                                            .43589, .9, outside ? 2048.5 : 14.5);
                        // The official CPU linear implementation's fixed-point block
                        // partition uses output width; pad before cropping as in warpReference.
                        const int referenceWidth = mode == INTER_CUBIC ? size.width
                            : ((size.width + 127) / 128) * 128;
                        Mat wide(size.height, referenceWidth, type, Scalar::all(37));
                        cv::warpAffine(host, wide, matrix, wide.size(), mode | WARP_INVERSE_MAP,
                                       border, value);
                        const Mat ref = wide(Rect(Point(), size)).clone();
                        const auto address = dst.address();
                        device::warpAffine(src, dst, matrix, size, mode | WARP_INVERSE_MAP,
                                           border, value, stream);
                        stream.wait();
                        ASSERT_EQ(address, dst.address());
                        const Mat got = dst.download();
                        if (depth != CV_32F)
                            EXPECT_LE(cv::norm(ref, got, NORM_INF), depth == CV_8U ? 0 : 1);
                        else
                            for (int y = 0; y < got.rows; ++y)
                                for (int x = 0; x < got.cols * channels; ++x)
                                    EXPECT_LE(std::abs(double(ref.ptr<float>(y)[x]) -
                                                       got.ptr<float>(y)[x]),
                                              warpF32Tolerance(ref.ptr<float>(y)[x]));
                        EXPECT_TRUE(guardsEqual(initial, target.download(), region));
                    }
                }
}

TEST(Triton_Common, FloatAlphaBitsAndPitchedOutput)
{
    requireDevice();
    device::Stream stream;
    for (const Size size : {Size(255, 257), Size(256, 256), Size(263, 257)})
    {
        SCOPED_TRACE(cv::format("width=%d height=%d", size.width, size.height));
        Mat host = randomImage(size, CV_32FC3, 75401);
        const std::uint32_t values[] = {0, 0x80000000u, 0x7f800000u, 0xff800000u,
                                   0x7fc12345u, 0x00000001u};
        std::memcpy(host.data, values, sizeof(values));
        Mat parent(size + Size(8, 2), host.type(), Scalar::all(17));
        const Rect sourceRegion(Point(1, 1), size);
        host.copyTo(parent(sourceRegion));
        device::Image storage;
        storage.upload(parent);
        device::Image src(storage, sourceRegion);
        const Mat initial(size + Size(8, 2), CV_32FC4, Scalar::all(37));
        device::Image target;
        target.upload(initial);
        const Rect region(Point(1, 1), size);
        device::Image dst(target, region);
        const auto address = dst.address();
        device::cvtColor(src, dst, COLOR_BGR2BGRA, stream);
        stream.wait();
        ASSERT_EQ(address, dst.address());
        const Mat got = dst.download();
        const Mat ref = cpuCvt(host, COLOR_BGR2BGRA);
        for (int y = 0; y < size.height; ++y)
            EXPECT_EQ(0, std::memcmp(got.ptr(y), ref.ptr(y), size.width * ref.elemSize()));
        EXPECT_TRUE(guardsEqual(initial, target.download(), region));
    }
}

TEST(Triton_Common, LuvInverseLightnessChromaPairs)
{
    requireDevice();
    device::Stream stream;
    // Every (L, u) pair; three v planes distinguish clipping and interior output.
    for (int v : {0, 127, 255})
        for (int code : {COLOR_Luv2BGR, COLOR_Luv2RGB, COLOR_Luv2LBGR, COLOR_Luv2LRGB})
        {
            SCOPED_TRACE(cv::format("v=%d code=%d", v, code));
            Mat host(256, 256, CV_8UC3);
            for (int l = 0; l < 256; ++l)
                for (int u = 0; u < 256; ++u)
                    host.at<Vec3b>(u, l) = Vec3b(l, u, v);
            device::Image src, dst;
            src.upload(host);
            device::cvtColor(src, dst, code, stream);
            stream.wait();
            EXPECT_EQ(0, cv::norm(cpuCvt(host, code), dst.download(), NORM_INF));
        }
}

TEST(Triton_Common, LuvInverseByteDomainAndPitchedOutput)
{
    requireDevice();
    device::Stream stream;
    for (int l : {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 191, 192, 254, 255})
        for (int code : {COLOR_Luv2BGR, COLOR_Luv2RGB, COLOR_Luv2LBGR, COLOR_Luv2LRGB})
        {
            SCOPED_TRACE(cv::format("L=%d code=%d", l, code));
            const Size size(257, 256);
            Mat host(size, CV_8UC3);
            for (int y = 0; y < size.height; ++y)
                for (int x = 0; x < size.width; ++x)
                    host.at<Vec3b>(y, x) = Vec3b(l, x & 255, y);
            Mat parent(size + Size(8, 2), host.type(), Scalar::all(17));
            const Rect sourceRegion(Point(1, 1), size);
            host.copyTo(parent(sourceRegion));
            device::Image storage;
            storage.upload(parent);
            device::Image src(storage, sourceRegion);
            const Mat initial(size + Size(8, 2), host.type(), Scalar::all(37));
            device::Image target;
            target.upload(initial);
            const Rect region(Point(1, 1), size);
            device::Image dst(target, region);
            device::cvtColor(src, dst, code, stream);
            stream.wait();
            EXPECT_EQ(0, cv::norm(cpuCvt(host, code), dst.download(), NORM_INF));
            EXPECT_TRUE(guardsEqual(initial, target.download(), region));
        }
}

TEST(Triton_Common, LargeChannelInterpolationTail)
{
    requireDevice();
    device::Stream stream;
    const Size size(2049, 2049);
    const Mat host = randomImage(size, CV_32FC4, 75291);
    const Mat matrix = (Mat_<double>(2, 3) << .9, -.43589, -23.5, .43589, .9, 14.5);
    const Mat ref = warpReference(host, matrix, size, INTER_LINEAR | WARP_INVERSE_MAP,
                                  BORDER_REPLICATE, Scalar());
    device::Image src;
    src.upload(host);
    const Mat initial(size + Size(8, 2), host.type(), Scalar::all(37));
    device::Image storage;
    storage.upload(initial);
    const Rect region(Point(1, 1), size);
    device::Image dst(storage, region);
    const auto address = dst.address();
    device::warpAffine(src, dst, matrix, size, INTER_LINEAR | WARP_INVERSE_MAP,
                       BORDER_REPLICATE, Scalar(), stream);
    stream.wait();
    ASSERT_EQ(address, dst.address());
    const Mat got = dst.download();
    for (int y = 0; y < got.rows; ++y)
        for (int x = 0; x < got.cols * got.channels(); ++x)
            ASSERT_LE(std::abs(double(ref.ptr<float>(y)[x]) - got.ptr<float>(y)[x]),
                      warpF32Tolerance(ref.ptr<float>(y)[x]));
    EXPECT_TRUE(guardsEqual(initial, storage.download(), region));
}

TEST(Triton_Common, LuvTableCornersAndPitchedOutput)
{
    requireDevice();
    device::Stream stream;
    const Size size(256, 256);
    for (int fixed : {0, 7, 8, 15, 16, 247, 248, 255})
        for (int code : {COLOR_BGR2Luv, COLOR_RGB2Luv})
        {
            SCOPED_TRACE(cv::format("fixed=%d code=%d", fixed, code));
            Mat parent(size + Size(8, 2), CV_8UC3, Scalar::all(17));
            const Rect region(Point(1, 1), size);
            Mat host = parent(region);
            for (int y = 0; y < host.rows; ++y)
                for (int x = 0; x < host.cols; ++x)
                    host.at<Vec3b>(y, x) = Vec3b(uchar(x), uchar(y), uchar(fixed));
            const Mat ref = cpuCvt(host, code);
            device::Image sourceStorage;
            sourceStorage.upload(parent);
            device::Image src(sourceStorage, region);
            const Mat initial(parent.size(), CV_8UC3, Scalar::all(37));
            device::Image target;
            target.upload(initial);
            device::Image dst(target, region);
            const auto address = dst.address();
            device::cvtColor(src, dst, code, stream);
            stream.wait();
            ASSERT_EQ(address, dst.address());
            EXPECT_EQ(0, cv::norm(ref, dst.download(), NORM_INF));
            EXPECT_TRUE(guardsEqual(initial, target.download(), region));
        }
}

TEST(Triton_Common, CubicRowReuse)
{
    requireDevice();
    device::Stream stream;
    uint64 seed = 75301;
    for (int width : {1023, 1024, 1025})
        for (int depth : {CV_8U, CV_32F})
            for (int channels : {1, 3, 4})
                for (int sourceHeight : {386, 514, 771})
                {
                    SCOPED_TRACE(cv::format("width=%d depth=%d cn=%d sourceHeight=%d", width,
                                            depth, channels, sourceHeight));
                    const Size size(width, 257);
                    const Mat host = randomImage(Size(width * 2 + 1, sourceHeight),
                                                  CV_MAKETYPE(depth, channels), seed++);
                    Mat parent(host.size() + Size(8, 2), host.type(), Scalar::all(17));
                    const Rect sourceRegion(Point(1, 1), host.size());
                    host.copyTo(parent(sourceRegion));
                    device::Image sourceStorage;
                    sourceStorage.upload(parent);
                    device::Image src(sourceStorage, sourceRegion);
                    const Mat initial(size + Size(8, 2), host.type(), Scalar::all(37));
                    device::Image target;
                    target.upload(initial);
                    const Rect region(Point(1, 1), size);
                    device::Image dst(target, region);
                    const Mat ref = oracle(host, size, 0, 0, INTER_CUBIC);
                    device::resize(src, dst, size, 0, 0, INTER_CUBIC, stream);
                    stream.wait();
                    EXPECT_TRUE(meetsContract(dst.download(), ref, INTER_CUBIC));
                    EXPECT_TRUE(guardsEqual(initial, target.download(), region));
                }
}

TEST(Triton_Common, WarpModesDepthsBordersAndLayouts)
{
    requireDevice();
    uint64 seed = 75001;
    Mat matrix = Mat::eye(2, 3, CV_64F);
    matrix.at<double>(0, 1) = 0.2;
    matrix.at<double>(0, 2) = 0.5;
    matrix.at<double>(1, 2) = -0.5;
    for (bool useDefault : {false, true})
    {
        device::Stream stream(useDefault);
        for (int depth : {CV_8U, CV_16U, CV_32F})
            for (int channels : {1, 3, 4})
                for (int mode : {INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA})
                    for (int border : {BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT,
                                       BORDER_WRAP, BORDER_REFLECT_101})
                        for (int inverse : {0, int(WARP_INVERSE_MAP)})
                        {
                            SCOPED_TRACE(
                                cv::format("depth=%d cn=%d mode=%d border=%d inverse=%d default=%d",
                                           depth, channels, mode, border, inverse, useDefault));
                            const Mat host =
                                randomImage(Size(17, 13), CV_MAKETYPE(depth, channels), seed++);
                            const Scalar value(0.5, 11, 23, 47);
                            const Mat ref = warpReference(host, matrix, Size(19, 11),
                                                          mode | inverse, border, value);
                            Mat parent(host.size() + Size(9, 5), host.type(), Scalar::all(17));
                            host.copyTo(parent(Rect(Point(1, 2), host.size())));
                            device::Image sourceStorage;
                            sourceStorage.upload(parent);
                            device::Image src(sourceStorage, Rect(Point(1, 2), host.size()));
                            const Mat initial(ref.size() + Size(9, 7), host.type(),
                                              Scalar::all(37));
                            device::Image target;
                            target.upload(initial);
                            const Rect rect(Point(2, 3), ref.size());
                            device::Image dst(target, rect);
                            const auto address = dst.address();
                            device::warpAffine(src, dst, matrix, ref.size(), mode | inverse, border,
                                               value, stream);
                            stream.wait();
                            ASSERT_EQ(address, dst.address());
                            const Mat got = dst.download();
                            for (int y = 0; y < got.rows; ++y)
                                for (int x = 0; x < got.cols * channels; ++x)
                                {
                                    if (depth == CV_8U)
                                        EXPECT_EQ(ref.ptr<uchar>(y)[x], got.ptr<uchar>(y)[x]);
                                    else if (depth == CV_16U)
                                        EXPECT_LE(std::abs(int(ref.ptr<ushort>(y)[x]) -
                                                           got.ptr<ushort>(y)[x]),
                                                  1);
                                    else
                                        EXPECT_LE(std::abs(double(ref.ptr<float>(y)[x]) -
                                                           got.ptr<float>(y)[x]),
                                                  warpF32Tolerance(ref.ptr<float>(y)[x]));
                                }
                            EXPECT_TRUE(guardsEqual(initial, target.download(), rect));
                        }
    }
}

TEST(Triton_Common, RejectionsPreserveExistingOutput)
{
    requireDevice();
    device::Stream stream;
    const Mat host(24, 32, CV_8UC3, Scalar(11, 23, 47));
    device::Image src;
    src.upload(host);
    const Mat old(7, 9, CV_8UC1, Scalar(63));
    device::Image dst;
    dst.upload(old);
    const auto address = dst.address();
    EXPECT_THROW(device::resize(src, dst, Size(16, 12), 0, 0, INTER_LANCZOS4, stream),
                 cv::Exception);
    EXPECT_THROW(device::resize(src, dst, Size(), std::numeric_limits<double>::infinity(), 1.0,
                                INTER_LINEAR, stream),
                 cv::Exception);
    EXPECT_THROW(device::resize(src, dst, Size(), 1e100, 1e100, INTER_LINEAR, stream),
                 cv::Exception);
    EXPECT_THROW(device::cvtColor(src, dst, COLOR_HSV2BGR, stream), cv::Exception);
    const Mat identity = Mat::eye(2, 3, CV_64F);
    EXPECT_THROW(device::warpAffine(src, dst, identity, Size(16, 12), INTER_CUBIC,
                                    BORDER_TRANSPARENT, Scalar(), stream),
                 cv::Exception);
    EXPECT_THROW(device::warpAffine(src, dst, identity, Size(16, 12), INTER_LINEAR | (1 << 12),
                                    BORDER_CONSTANT, Scalar(), stream),
                 cv::Exception);
    Mat invalid = identity.clone();
    invalid.at<double>(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(device::warpAffine(src, dst, invalid, Size(16, 12), INTER_LINEAR, BORDER_CONSTANT,
                                    Scalar(), stream),
                 cv::Exception);
    invalid.at<double>(0, 0) = 1e100;
    EXPECT_THROW(device::warpAffine(src, dst, invalid, Size(16, 12),
                                    INTER_LINEAR | WARP_INVERSE_MAP, BORDER_CONSTANT, Scalar(),
                                    stream),
                 cv::Exception);
    EXPECT_THROW(device::warpAffine(src, dst, identity, Size(16, 12), INTER_LINEAR, BORDER_CONSTANT,
                                    Scalar(std::numeric_limits<double>::quiet_NaN()), stream),
                 cv::Exception);
    EXPECT_EQ(address, dst.address());
    EXPECT_EQ(0, cv::norm(old, dst.download(), NORM_INF));
    EXPECT_THROW(device::resize(src, src, Size(16, 12), 0, 0, INTER_LINEAR, stream), cv::Exception);
    EXPECT_THROW(device::cvtColor(src, src, COLOR_BGR2GRAY, stream), cv::Exception);
    EXPECT_THROW(device::warpAffine(src, src, identity, Size(16, 12), INTER_LINEAR, BORDER_CONSTANT,
                                    Scalar(), stream),
                 cv::Exception);
    EXPECT_EQ(0, cv::norm(host, src.download(), NORM_INF));
    // Explicit dsize wins: ignored fx/fy must not influence either validation or numerics.
    device::resize(src, dst, Size(16, 12), std::numeric_limits<double>::quiet_NaN(),
                   std::numeric_limits<double>::infinity(), INTER_LINEAR, stream);
    stream.wait();
    EXPECT_TRUE(meetsContract(dst.download(), oracle(host, Size(16, 12), 0, 0, INTER_LINEAR),
                              INTER_LINEAR));
}

TEST(Triton_Common, InvalidDepthAndOffsetOverlapPreserveOutput)
{
    requireDevice();
    device::Stream stream;
    const Mat identity = Mat::eye(2, 3, CV_64F);
    const Mat initial(7, 9, CV_8UC1, Scalar(63));
    device::Image output;
    output.upload(initial);
    const auto address = output.address();
    for (int depth : {CV_16S, CV_64F})
    {
        device::Image source;
        source.upload(Mat(24, 32, CV_MAKETYPE(depth, 3), Scalar::all(11)));
        EXPECT_THROW(device::resize(source, output, Size(16, 12), 0, 0, INTER_LINEAR, stream),
                     cv::Exception);
        EXPECT_THROW(device::cvtColor(source, output, COLOR_BGR2GRAY, stream), cv::Exception);
        EXPECT_THROW(device::warpAffine(source, output, identity, Size(16, 12), INTER_LINEAR,
                                        BORDER_REPLICATE, Scalar(), stream),
                     cv::Exception);
        EXPECT_EQ(address, output.address());
        EXPECT_EQ(0, cv::norm(initial, output.download(), NORM_INF));
    }
    const Mat parent(32, 48, CV_8UC3, Scalar(11, 23, 47));
    device::Image storage;
    storage.upload(parent);
    device::Image source(storage, Rect(1, 1, 32, 24)), overlapping(storage, Rect(2, 2, 16, 12));
    const auto overlapAddress = overlapping.address();
    const Mat overlapInitial = overlapping.download();
    EXPECT_THROW(device::resize(source, overlapping, Size(15, 11), 0, 0, INTER_LINEAR, stream),
                 cv::Exception);
    EXPECT_THROW(device::cvtColor(source, overlapping, COLOR_BGR2GRAY, stream), cv::Exception);
    EXPECT_THROW(device::warpAffine(source, overlapping, identity, Size(15, 11), INTER_LINEAR,
                                    BORDER_REPLICATE, Scalar(), stream),
                 cv::Exception);
    EXPECT_EQ(overlapAddress, overlapping.address());
    EXPECT_EQ(0, cv::norm(overlapInitial, overlapping.download(), NORM_INF));
    EXPECT_EQ(0, cv::norm(parent, storage.download(), NORM_INF));
}

// Element-aligned C4 views may start inside a pixel. The public descriptor
// exercises those offsets and the packed-word fallback.
TEST(Triton_Common, NearestWordAlignment)
{
    requireDevice();
    if (std::strcmp(device::backend(), "hip") != 0)
        throw SkipTestException("HIP descriptor alignment coverage");
    device::Stream stream(true);
    const Mat matrix = (Mat_<double>(2, 3) << .9, -.43589, -23.5, .43589, .9, 14.5);
    for (int type : {CV_16UC4, CV_32FC4})
        for (int width : {255, 256, 257})
            for (std::size_t offset : {std::size_t(0), std::size_t(CV_ELEM_SIZE1(type)), std::size_t(8)})
            {
                SCOPED_TRACE(cv::format("type=%d width=%d byte offset=%zu", type, width, offset));
                const Size size(width, 256), parentSize(width + 4, 258);
                Mat host = randomImage(parentSize, type, 78201);
                if (CV_MAT_DEPTH(type) == CV_32F)
                {
                    const std::uint32_t bits[] = {0x80000000u, 1u, 0x7f800000u, 0x7fc12345u};
                    for (int y = 1; y <= size.height; ++y)
                        std::memcpy(host.ptr(y) + offset, bits, sizeof(bits));
                }
                const Mat source(size, type,
                                 const_cast<uchar*>(host.ptr(1)) + offset, host.step);
                const Mat expected = warpReference(source, matrix, size,
                    INTER_NEAREST | WARP_INVERSE_MAP, BORDER_REPLICATE);
                const Mat initial(parentSize, type, Scalar::all(73));
                device::Image input, output;
                input.upload(host);
                output.upload(initial);
                cv::triton::DeviceImage src, dst;
                src.ptr = input.address() + input.step() + offset;
                src.rows = dst.rows = size.height;
                src.cols = dst.cols = size.width;
                src.type = dst.type = type;
                src.step = input.step();
                dst.ptr = output.address() + output.step() + offset;
                dst.step = output.step();
                cv::triton::warpAffine(src, dst, matrix.ptr<double>(),
                    INTER_NEAREST | WARP_INVERSE_MAP, BORDER_REPLICATE, Scalar(), 0, 0);
                stream.wait();
                const Mat result = output.download();
                for (int y = 0; y < parentSize.height; ++y)
                {
                    const bool inside = y >= 1 && y <= size.height;
                    const std::size_t begin = inside ? offset : 0;
                    const std::size_t end = inside ? offset + size.width * result.elemSize() : 0;
                    EXPECT_EQ(0, std::memcmp(initial.ptr(y), result.ptr(y), begin));
                    EXPECT_EQ(0, std::memcmp(initial.ptr(y) + end, result.ptr(y) + end,
                                             parentSize.width * result.elemSize() - end));
                    if (inside)
                    {
                        EXPECT_EQ(0, std::memcmp(expected.ptr(y - 1), result.ptr(y) + begin,
                                                 size.width * result.elemSize()));
                    }
                }
            }
}

TEST(Triton_Common, HalfPixelCubicPitchedOutput)
{
    requireDevice();
    device::Stream stream;
    const Size size(257, 259);
    for (int type : {CV_32FC1, CV_8UC4, CV_16UC4})
        for (double dx : {-.5, .5})
            for (double dy : {-.5, .5})
            {
                SCOPED_TRACE(cv::format("type=%d dx=%g dy=%g", type, dx, dy));
                Mat host = randomImage(size, type, 79501);
                if (CV_MAT_DEPTH(type) == CV_32F)
                    host = host * 4.f - 2.f;
                const Mat matrix = (Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
                const Mat expected = warpReference(host, matrix, size,
                    INTER_CUBIC | WARP_INVERSE_MAP, BORDER_REPLICATE);
                Mat parent(size + Size(8, 2), type, Scalar::all(17));
                const Rect region(Point(1, 1), size);
                host.copyTo(parent(region));
                device::Image storage;
                storage.upload(parent);
                device::Image src(storage, region);
                const Mat initial(parent.size(), type, Scalar::all(73));
                device::Image target;
                target.upload(initial);
                device::Image dst(target, region);
                device::warpAffine(src, dst, matrix, size, INTER_CUBIC | WARP_INVERSE_MAP,
                                   BORDER_REPLICATE, Scalar(), stream);
                stream.wait();
                const Mat got = dst.download();
                if (CV_MAT_DEPTH(type) == CV_32F)
                    for (int y = 0; y < size.height; ++y)
                        for (int x = 0; x < size.width; ++x)
                            EXPECT_LE(std::abs(double(got.ptr<float>(y)[x]) - expected.ptr<float>(y)[x]),
                                      warpF32Tolerance(expected.ptr<float>(y)[x]));
                else
                    EXPECT_LE(cv::norm(got, expected, NORM_INF), CV_MAT_DEPTH(type) == CV_8U ? 0. : 1.);
                EXPECT_TRUE(guardsEqual(initial, target.download(), region));
            }
}

TEST(Triton_Common, IntegerIdentityInterpolation)
{
    requireDevice();
    device::Stream stream;
    const Size size(512, 512);
    const Mat matrix = Mat::eye(2, 3, CV_64F);
    for (int depth : {CV_8U, CV_16U})
        for (int cn : {1, 3, 4})
        {
            const int type = CV_MAKETYPE(depth, cn);
            Mat host(size, type);
            for (std::size_t i = 0; i < host.total() * cn; ++i)
                if (depth == CV_8U)
                    host.ptr<uchar>()[i] = static_cast<uchar>(i);
                else
                    host.ptr<ushort>()[i] = static_cast<ushort>(i);
            Mat parent(size + Size(32, 2), type, Scalar::all(17));
            const Rect region(Point(16, 1), size);
            host.copyTo(parent(region));
            device::Image storage;
            storage.upload(parent);
            device::Image src(storage, region);
            for (int mode : {INTER_LINEAR, INTER_CUBIC})
                for (int border : {BORDER_CONSTANT, BORDER_TRANSPARENT})
                {
                    SCOPED_TRACE(cv::format("depth=%d cn=%d mode=%d border=%d",
                                            depth, cn, mode, border));
                    const Mat initial(parent.size(), type, Scalar::all(37));
                    device::Image target;
                    target.upload(initial);
                    device::Image dst(target, region);
                    const auto address = dst.address();
                    if (mode == INTER_CUBIC && border == BORDER_TRANSPARENT)
                    {
                        EXPECT_THROW(device::warpAffine(src, dst, matrix, size,
                            mode | WARP_INVERSE_MAP, border, Scalar::all(113), stream), cv::Exception);
                        stream.wait();
                        EXPECT_EQ(0, cv::norm(target.download(), initial, NORM_INF));
                        continue;
                    }
                    const Mat reference = warpReference(host, matrix, size,
                        mode | WARP_INVERSE_MAP, border, Scalar::all(113));
                    device::warpAffine(src, dst, matrix, size, mode | WARP_INVERSE_MAP,
                                       border, Scalar::all(113), stream);
                    stream.wait();
                    EXPECT_EQ(address, dst.address());
                    EXPECT_EQ(0, cv::norm(reference, host, NORM_INF));
                    EXPECT_EQ(0, cv::norm(dst.download(), reference, NORM_INF));
                    EXPECT_TRUE(guardsEqual(initial, target.download(), region));
                }
        }
}

TEST(Triton_Common, CubicPitchedOffsets)
{
    requireDevice();
    if (std::strcmp(device::backend(), "hip") != 0)
        throw SkipTestException("HIP descriptor alignment coverage");
    device::Stream stream(true);
    const Mat matrix = (Mat_<double>(2, 3) << .9, -.43589, -23.5, .43589, .9, 14.5);
    for (int type : {CV_8UC4, CV_16UC4})
        for (int width : {255, 256, 257})
            for (std::size_t offset : {std::size_t(0), std::size_t(CV_ELEM_SIZE1(type)),
                                       std::size_t(3 * CV_ELEM_SIZE1(type)), std::size_t(CV_ELEM_SIZE(type))})
            {
                SCOPED_TRACE(cv::format("type=%d width=%d offset=%zu", type, width, offset));
                const Size size(width, 256), parentSize(width + 8, 258);
                Mat host = randomImage(parentSize, type, 79401);
                const Mat source(size, type, host.ptr(1) + offset, host.step);
                const Mat expected = warpReference(source, matrix, size,
                    INTER_CUBIC | WARP_INVERSE_MAP, BORDER_REPLICATE);
                const Mat initial(parentSize, type, Scalar::all(73));
                device::Image input, output;
                input.upload(host);
                output.upload(initial);
                cv::triton::DeviceImage src, dst;
                src.ptr = input.address() + input.step() + offset;
                src.rows = dst.rows = size.height;
                src.cols = dst.cols = size.width;
                src.type = dst.type = type;
                src.step = input.step();
                dst.ptr = output.address() + output.step() + offset;
                dst.step = output.step();
                cv::triton::warpAffine(src, dst, matrix.ptr<double>(),
                    INTER_CUBIC | WARP_INVERSE_MAP, BORDER_REPLICATE, Scalar(), 0, 0);
                stream.wait();
                Mat result = output.download();
                const Mat got(size, type, result.ptr(1) + offset, result.step);
                EXPECT_LE(cv::norm(got, expected, NORM_INF), CV_MAT_DEPTH(type) == CV_8U ? 0. : 1.);
                for (int y = 0; y < parentSize.height; ++y)
                {
                    const bool inside = y >= 1 && y <= size.height;
                    const std::size_t begin = inside ? offset : 0;
                    const std::size_t end = inside ? offset + size.width * result.elemSize() : 0;
                    EXPECT_EQ(0, std::memcmp(initial.ptr(y), result.ptr(y), begin));
                    EXPECT_EQ(0, std::memcmp(initial.ptr(y) + end, result.ptr(y) + end,
                                             parentSize.width * result.elemSize() - end));
                }
            }
}

TEST(Triton_Common, FloatGrayC4Layouts)
{
    requireDevice();
    if (std::strcmp(device::backend(), "hip") != 0)
        throw SkipTestException("HIP descriptor alignment coverage");
    struct RestoreRounding
    {
        int mode = std::fegetround();
        ~RestoreRounding() { std::fesetround(mode); }
    } restore;
    device::Stream stream(true);
    for (int rounding : {FE_TONEAREST, FE_DOWNWARD})
    {
        if (std::fesetround(rounding) != 0)
            continue;
        if (rounding == FE_DOWNWARD)
            EXPECT_FALSE(cv_triton_color::float_cpu_separate4());
        for (int width : {255, 256, 257})
            for (int code : {COLOR_BGRA2GRAY, COLOR_RGBA2GRAY})
                for (std::size_t offset : {std::size_t(0), std::size_t(4), std::size_t(12)})
                {
                    SCOPED_TRACE(cv::format("round=%d width=%d code=%d offset=%zu",
                                            rounding, width, code, offset));
                    const Size size(width, 256), parentSize(width + 8, 258);
                    Mat host = randomImage(parentSize, CV_32FC4, 79301);
                    for (int y = 0; y < host.rows; ++y)
                        for (int x = 0; x < host.cols * 4; ++x)
                            host.ptr<float>(y)[x] *= x % 3 ? 32767.5f : -1.f;
                    const Mat source(size, CV_32FC4, host.ptr(1) + offset, host.step);
                    const Mat expected = cpuCvt(source, code);
                    const Mat initial(parentSize, CV_32FC1, Scalar::all(73));
                    device::Image input, output;
                    input.upload(host);
                    output.upload(initial);
                    cv::triton::DeviceImage src, dst;
                    src.ptr = input.address() + input.step() + offset;
                    src.rows = dst.rows = size.height;
                    src.cols = dst.cols = size.width;
                    src.type = CV_32FC4;
                    src.step = input.step();
                    dst.ptr = output.address() + output.step() + offset;
                    dst.type = CV_32FC1;
                    dst.step = output.step();
                    cv::triton::cvtColor(src, dst, code, 0, 0);
                    stream.wait();
                    Mat result = output.download();
                    const Mat got(size, CV_32FC1, result.ptr(1) + offset, result.step);
                    EXPECT_TRUE(meetsCvtContract(got, expected, source, true,
                                                 cv_triton_color::float_cpu_separate4()));
                    for (int y = 0; y < parentSize.height; ++y)
                    {
                        const bool inside = y >= 1 && y <= size.height;
                        const std::size_t begin = inside ? offset : 0;
                        const std::size_t end = inside ? offset + size.width * sizeof(float) : 0;
                        EXPECT_EQ(0, std::memcmp(initial.ptr(y), result.ptr(y), begin));
                        EXPECT_EQ(0, std::memcmp(initial.ptr(y) + end, result.ptr(y) + end,
                                                 parentSize.width * sizeof(float) - end));
                    }
                }
    }
}

TEST(Triton_Common, ResizeCubicC4Enlargement)
{
    requireDevice();
    device::Stream stream;
    for (int depth : {CV_8U, CV_32F})
        for (Size inputSize : {Size(131, 129), Size(389, 139)})
        {
            const Size outputSize(263, 281);
            const Mat host = randomImage(inputSize, CV_MAKETYPE(depth, 4), 78301);
            const Mat expected = oracle(host, outputSize, 0, 0, INTER_CUBIC);
            Mat parent(inputSize + Size(8, 2), host.type(), Scalar::all(17));
            host.copyTo(parent(Rect(Point(1, 1), inputSize)));
            device::Image storage;
            storage.upload(parent);
            device::Image src(storage, Rect(Point(1, 1), inputSize));
            const Mat initial(outputSize + Size(8, 2), host.type(), Scalar::all(73));
            device::Image target;
            target.upload(initial);
            const Rect roi(Point(3, 1), outputSize);
            device::Image dst(target, roi);
            device::resize(src, dst, outputSize, 0, 0, INTER_CUBIC, stream);
            stream.wait();
            EXPECT_TRUE(meetsContract(dst.download(), expected, INTER_CUBIC));
            EXPECT_TRUE(guardsEqual(initial, target.download(), roi));
        }
}

// Identity nearest must preserve pixel bits and the destination's surrounding storage.
// Even/odd widths and offset views exercise both the copy path and its fallback.
TEST(Triton_Common, IdentityNearestCopyLayouts)
{
    requireDevice();
    device::Stream stream;
    const Mat identity = Mat::eye(2, 3, CV_64F);
    for (int depth : {CV_8U, CV_16U, CV_32F})
        for (int channels : {1, 3, 4})
            for (Size size : {Size(512, 512), Size(513, 513),
                              Size(256, 1024), Size(255, 1029)})
                for (int offset : {0, 1})
                {
                    const int type = CV_MAKETYPE(depth, channels);
                    SCOPED_TRACE(cv::format("depth=%d channels=%d size=%dx%d offset=%d",
                                            depth, channels, size.width, size.height, offset));
                    const Size storageSize(size.width + 16, size.height + 2);
                    Mat host = randomImage(storageSize, type, 78101);
                    const Rect roi(offset, 1, size.width, size.height);
                    if (depth == CV_32F)
                    {
                        // Include signed zero, subnormal, infinity and a quiet NaN payload.
                        const std::uint32_t bits[] = {0x80000000u, 1u, 0x7f800000u, 0x7fc12345u};
                        std::memcpy(host(roi).ptr(), bits, sizeof(bits));
                    }
                    device::Image inputStorage, outputStorage;
                    inputStorage.upload(host);
                    const Mat initial(storageSize, type, Scalar::all(73));
                    for (int border : {BORDER_REPLICATE, BORDER_TRANSPARENT})
                    {
                        outputStorage.upload(initial);
                        device::Image src(inputStorage, roi), dst(outputStorage, roi);
                        device::warpAffine(src, dst, identity, size, INTER_NEAREST,
                                           border, Scalar(), stream);
                        stream.wait();
                        const Mat result = outputStorage.download();
                        EXPECT_TRUE(guardsEqual(initial, result, roi));
                        for (int y = 0; y < size.height; ++y)
                            ASSERT_EQ(0, std::memcmp(host(roi).ptr(y), result(roi).ptr(y),
                                                     size.width * host.elemSize()));
                    }
                }
}

TEST(Triton_Common, ThreeOperatorPipelineOnCallerStream)
{
    requireDevice();
    device::Stream stream;
    const Mat host = randomImage(Size(64, 48), CV_8UC3, 76001);
    const Mat identity = Mat::eye(2, 3, CV_64F);
    device::Image src;
    src.upload(host);
    device::Image resized, gray, warped;
    device::resize(src, resized, Size(32, 24), 0, 0, INTER_LINEAR, stream);
    device::cvtColor(resized, gray, COLOR_BGR2GRAY, stream);
    device::warpAffine(gray, warped, identity, Size(32, 24), INTER_NEAREST, BORDER_REPLICATE,
                       Scalar(), stream);
    stream.wait();
    EXPECT_EQ(0, cv::norm(warped.download(),
                          cpuCvt(oracle(host, Size(32, 24), 0, 0, INTER_LINEAR), COLOR_BGR2GRAY),
                          NORM_INF));
}
TEST(Triton_Common, CrossStreamProducerConsumer)
{
    requireDevice();
    device::Stream producer, consumer;
    const Mat host = randomImage(Size(96, 64), CV_8UC3, 77001);
    const Mat identity = Mat::eye(2, 3, CV_64F);
    device::Image src;
    src.upload(host);
    device::Image small, gray, output;
    // Preallocate and warm all keys outside the ordering-sensitive interval: module loading
    // may synchronize a context, which would hide a missing producer/consumer dependency.
    device::resize(src, small, Size(63, 47), 0, 0, INTER_AREA, producer);
    producer.wait();
    device::cvtColor(small, gray, COLOR_BGR2GRAY, consumer);
    consumer.wait();
    device::warpAffine(gray, output, identity, Size(63, 47), INTER_NEAREST, BORDER_REPLICATE,
                       Scalar(), consumer);
    consumer.wait();
    const Mat changed = randomImage(host.size(), host.type(), 77002);
    src.upload(changed);
    device::resize(src, small, Size(63, 47), 0, 0, INTER_AREA, producer);
    consumer.waitFor(producer);
    device::cvtColor(small, gray, COLOR_BGR2GRAY, consumer);
    device::warpAffine(gray, output, identity, Size(63, 47), INTER_NEAREST, BORDER_REPLICATE,
                       Scalar(), consumer);
    consumer.wait();
    EXPECT_EQ(0, cv::norm(output.download(),
                          cpuCvt(oracle(changed, Size(63, 47), 0, 0, INTER_AREA), COLOR_BGR2GRAY),
                          NORM_INF));
}

}
}
