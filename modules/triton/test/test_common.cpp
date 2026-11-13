// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Shared accuracy through real public APIs, without CUDA/HIP SDK headers.
#include "../validation/device.hpp"
#include <cstring>
#include <limits>
#include "../validation/reference.hpp"
#include "../validation/resize_cases.hpp"
#include "../validation/cvtcolor_cases.hpp"

namespace opencv_test { namespace {
namespace device = triton_validation;
using namespace triton_test;

static void requireDevice()
{
    if (!device::available()) throw SkipTestException("no usable validation backend");
}

static bool guardsEqual(const Mat& before, const Mat& after, Rect changed)
{
    if (before.size() != after.size() || before.type() != after.type()) return false;
    for (int y = 0; y < before.rows; ++y)
        for (int x = 0; x < before.cols; ++x)
            if (!changed.contains(Point(x, y)) &&
                std::memcmp(before.ptr(y) + x * before.elemSize(),
                            after.ptr(y) + x * after.elemSize(), before.elemSize()) != 0) return false;
    return true;
}

TEST(Triton_Common, ResizeRoutesLayoutsAndStreams)
{
    requireDevice();
    uint64 seed = 73001;
    for (bool useDefault : {false, true})
    {
        device::Stream stream(useDefault);
        for (const auto& c : resize_cases::kCases)
            for (int type : resize_cases::kTypes)
            {
                SCOPED_TRACE(cv::format("backend=%s mode=%d type=%d default=%d", device::backend(), c.interp, type, useDefault));
                const Mat host = randomImage(c.src, type, seed++);
                const Mat reference = oracle(host, c.dst, c.fx, c.fy, c.interp);
                Mat sourceParent(c.src + Size(13, 7), type, Scalar::all(23));
                host.copyTo(sourceParent(Rect(Point(1, 2), c.src)));
                device::Image sourceStorage; sourceStorage.upload(sourceParent);
                device::Image src(sourceStorage, Rect(Point(1, 2), c.src));
                const Mat initial(reference.size() + Size(17, 9), type, Scalar::all(37));
                device::Image targetStorage; targetStorage.upload(initial);
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

static void checkCvt(const cvt_cases::CvtCase& c, uint64 seed, device::Stream& stream)
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
    device::Image storage; storage.upload(parent);
    device::Image src(storage, Rect(Point(2, 1), size));
    const Mat initial(reference.size() + Size(13, 8), reference.type(), Scalar::all(43));
    device::Image target; target.upload(initial);
    const Rect outputRect(Point(1, 3), reference.size());
    device::Image dst(target, outputRect);
    const auto address = dst.address();
    device::cvtColor(src, dst, c.code, stream);
    stream.wait();
    ASSERT_EQ(address, dst.address());
    const Mat got = dst.download();
    ASSERT_EQ(reference.size(), got.size()); ASSERT_EQ(reference.type(), got.type());
    EXPECT_TRUE(meetsCvtContract(got, reference, host, cvt_cases::floatArithmetic(c.code)));
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
    // The neighbouring large pixel must not lend its tolerance to this one.
    got.at<float>(0, 0) = 0.001f;
    EXPECT_FALSE(meetsCvtContract(got, reference, input, true));
    EXPECT_FALSE(meetsCvtContract(got, reference, input, false));
    got = reference.clone(); got.at<float>(0, 0) = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(meetsCvtContract(got, reference, input, true));
    Mat exact(1, 2, CV_16UC1, Scalar::all(65535));
    Mat changed = exact.clone(); changed.at<ushort>(0, 0) = 65534;
    EXPECT_FALSE(meetsCvtContract(changed, exact, exact, false));
}

TEST(Triton_Common, CvtColorAllDeclaredCodesAndDepths)
{
    requireDevice();
    uint64 seed = 74001;
    for (bool useDefault : {false, true})
    {
        device::Stream stream(useDefault);
        for (const auto& c : cvt_cases::kCases) { SCOPED_TRACE(c.name); checkCvt(c, seed++, stream); }
        for (const auto& c : cvt_cases::kCases16U) { SCOPED_TRACE(c.name); checkCvt(c, seed++, stream); }
        for (const auto& c : cvt_cases::kCases32F) { SCOPED_TRACE(c.name); checkCvt(c, seed++, stream); }
    }
}

TEST(Triton_Common, WarpModesDepthsBordersAndLayouts)
{
    requireDevice();
    uint64 seed = 75001;
    Mat matrix = Mat::eye(2, 3, CV_64F);
    matrix.at<double>(0, 1) = 0.2; matrix.at<double>(0, 2) = 0.5; matrix.at<double>(1, 2) = -0.5;
    for (bool useDefault : {false, true})
    {
        device::Stream stream(useDefault);
        for (int depth : {CV_8U, CV_16U, CV_32F})
            for (int channels : {1, 3, 4})
                for (int mode : {INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA})
                    for (int border : {BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101})
                        for (int inverse : {0, int(WARP_INVERSE_MAP)})
                        {
                            SCOPED_TRACE(cv::format("depth=%d cn=%d mode=%d border=%d inverse=%d default=%d",
                                                   depth, channels, mode, border, inverse, useDefault));
                            const Mat host = randomImage(Size(17, 13), CV_MAKETYPE(depth, channels), seed++);
                            const Scalar value(0.5, 11, 23, 47);
                            const Mat ref = warpReference(host, matrix, Size(19, 11), mode | inverse, border, value);
                            Mat parent(host.size() + Size(9, 5), host.type(), Scalar::all(17));
                            host.copyTo(parent(Rect(Point(1, 2), host.size())));
                            device::Image sourceStorage; sourceStorage.upload(parent);
                            device::Image src(sourceStorage, Rect(Point(1, 2), host.size()));
                            const Mat initial(ref.size() + Size(9, 7), host.type(), Scalar::all(37));
                            device::Image target; target.upload(initial);
                            const Rect rect(Point(2, 3), ref.size());
                            device::Image dst(target, rect);
                            const auto address = dst.address();
                            device::warpAffine(src, dst, matrix, ref.size(), mode | inverse, border, value, stream);
                            stream.wait();
                            ASSERT_EQ(address, dst.address());
                            const Mat got = dst.download();
                            for (int y = 0; y < got.rows; ++y)
                                for (int x = 0; x < got.cols * channels; ++x)
                                {
                                    if (depth == CV_8U) EXPECT_EQ(ref.ptr<uchar>(y)[x], got.ptr<uchar>(y)[x]);
                                    else if (depth == CV_16U) EXPECT_LE(std::abs(int(ref.ptr<ushort>(y)[x]) - got.ptr<ushort>(y)[x]), 1);
                                    else EXPECT_LE(std::abs(double(ref.ptr<float>(y)[x]) - got.ptr<float>(y)[x]),
                                                   warpF32Tolerance(ref.ptr<float>(y)[x]));
                                }
                            EXPECT_TRUE(guardsEqual(initial, target.download(), rect));
                        }
    }
}

TEST(Triton_Common, RejectionsPreserveExistingOutput)
{
    requireDevice(); device::Stream stream;
    const Mat host(24, 32, CV_8UC3, Scalar(11, 23, 47));
    device::Image src; src.upload(host);
    const Mat old(7, 9, CV_8UC1, Scalar(63));
    device::Image dst; dst.upload(old);
    const auto address = dst.address();
    EXPECT_THROW(device::resize(src, dst, Size(16, 12), 0, 0, INTER_LANCZOS4, stream), cv::Exception);
    EXPECT_THROW(device::resize(src, dst, Size(), std::numeric_limits<double>::infinity(),
                                1.0, INTER_LINEAR, stream), cv::Exception);
    EXPECT_THROW(device::resize(src, dst, Size(), 1e100, 1e100, INTER_LINEAR, stream), cv::Exception);
    EXPECT_THROW(device::cvtColor(src, dst, COLOR_HSV2BGR, stream), cv::Exception);
    const Mat identity = Mat::eye(2, 3, CV_64F);
    EXPECT_THROW(device::warpAffine(src, dst, identity, Size(16, 12), INTER_CUBIC,
                                   BORDER_TRANSPARENT, Scalar(), stream), cv::Exception);
    EXPECT_THROW(device::warpAffine(src, dst, identity, Size(16, 12), INTER_LINEAR | (1 << 12),
                                   BORDER_CONSTANT, Scalar(), stream), cv::Exception);
    Mat invalid = identity.clone();
    invalid.at<double>(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(device::warpAffine(src, dst, invalid, Size(16, 12), INTER_LINEAR,
                                   BORDER_CONSTANT, Scalar(), stream), cv::Exception);
    invalid.at<double>(0, 0) = 1e100;
    EXPECT_THROW(device::warpAffine(src, dst, invalid, Size(16, 12), INTER_LINEAR | WARP_INVERSE_MAP,
                                   BORDER_CONSTANT, Scalar(), stream), cv::Exception);
    EXPECT_THROW(device::warpAffine(src, dst, identity, Size(16, 12), INTER_LINEAR,
                                   BORDER_CONSTANT, Scalar(std::numeric_limits<double>::quiet_NaN()), stream), cv::Exception);
    EXPECT_EQ(address, dst.address()); EXPECT_EQ(0, cv::norm(old, dst.download(), NORM_INF));
    EXPECT_THROW(device::resize(src, src, Size(16, 12), 0, 0, INTER_LINEAR, stream), cv::Exception);
    EXPECT_THROW(device::cvtColor(src, src, COLOR_BGR2GRAY, stream), cv::Exception);
    EXPECT_THROW(device::warpAffine(src, src, identity, Size(16, 12), INTER_LINEAR,
                                   BORDER_CONSTANT, Scalar(), stream), cv::Exception);
    EXPECT_EQ(0, cv::norm(host, src.download(), NORM_INF));
    // Explicit dsize wins: ignored fx/fy must not influence either validation or numerics.
    device::resize(src, dst, Size(16, 12), std::numeric_limits<double>::quiet_NaN(),
                   std::numeric_limits<double>::infinity(), INTER_LINEAR, stream);
    stream.wait();
    EXPECT_TRUE(meetsContract(dst.download(), oracle(host, Size(16, 12), 0, 0, INTER_LINEAR), INTER_LINEAR));

}

TEST(Triton_Common, InvalidDepthAndOffsetOverlapPreserveOutput)
{
    requireDevice(); device::Stream stream;
    const Mat identity=Mat::eye(2,3,CV_64F);
    const Mat initial(7,9,CV_8UC1,Scalar(63));
    device::Image output;output.upload(initial);
    const auto address=output.address();
    for (int depth : {CV_16S,CV_64F})
    {
        device::Image source;source.upload(Mat(24,32,CV_MAKETYPE(depth,3),Scalar::all(11)));
        EXPECT_THROW(device::resize(source,output,Size(16,12),0,0,INTER_LINEAR,stream),cv::Exception);
        EXPECT_THROW(device::cvtColor(source,output,COLOR_BGR2GRAY,stream),cv::Exception);
        EXPECT_THROW(device::warpAffine(source,output,identity,Size(16,12),INTER_LINEAR,
                                       BORDER_REPLICATE,Scalar(),stream),cv::Exception);
        EXPECT_EQ(address,output.address());EXPECT_EQ(0,cv::norm(initial,output.download(),NORM_INF));
    }
    const Mat parent(32,48,CV_8UC3,Scalar(11,23,47));
    device::Image storage;storage.upload(parent);
    device::Image source(storage,Rect(1,1,32,24)), overlapping(storage,Rect(2,2,16,12));
    const auto overlapAddress=overlapping.address();
    const Mat overlapInitial=overlapping.download();
    EXPECT_THROW(device::resize(source,overlapping,Size(15,11),0,0,INTER_LINEAR,stream),cv::Exception);
    EXPECT_THROW(device::cvtColor(source,overlapping,COLOR_BGR2GRAY,stream),cv::Exception);
    EXPECT_THROW(device::warpAffine(source,overlapping,identity,Size(15,11),INTER_LINEAR,
                                   BORDER_REPLICATE,Scalar(),stream),cv::Exception);
    EXPECT_EQ(overlapAddress,overlapping.address());
    EXPECT_EQ(0,cv::norm(overlapInitial,overlapping.download(),NORM_INF));
    EXPECT_EQ(0,cv::norm(parent,storage.download(),NORM_INF));
}

TEST(Triton_Common, ThreeOperatorPipelineOnCallerStream)
{
    requireDevice(); device::Stream stream;
    const Mat host = randomImage(Size(64, 48), CV_8UC3, 76001);
    const Mat identity = Mat::eye(2, 3, CV_64F);
    device::Image src; src.upload(host);
    device::Image resized, gray, warped;
    device::resize(src, resized, Size(32, 24), 0, 0, INTER_LINEAR, stream);
    device::cvtColor(resized, gray, COLOR_BGR2GRAY, stream);
    device::warpAffine(gray, warped, identity, Size(32, 24), INTER_NEAREST, BORDER_REPLICATE, Scalar(), stream);
    stream.wait();
    EXPECT_EQ(0, cv::norm(warped.download(), cpuCvt(oracle(host, Size(32, 24), 0, 0, INTER_LINEAR), COLOR_BGR2GRAY), NORM_INF));
}
TEST(Triton_Common, CrossStreamProducerConsumer)
{
    requireDevice(); device::Stream producer, consumer;
    const Mat host = randomImage(Size(96, 64), CV_8UC3, 77001);
    const Mat identity = Mat::eye(2, 3, CV_64F);
    device::Image src; src.upload(host);
    device::Image small, gray, output;
    // Preallocate and warm all keys outside the ordering-sensitive interval: module loading
    // may synchronize a context, which would hide a missing producer/consumer dependency.
    device::resize(src, small, Size(63, 47), 0, 0, INTER_AREA, producer); producer.wait();
    device::cvtColor(small, gray, COLOR_BGR2GRAY, consumer); consumer.wait();
    device::warpAffine(gray, output, identity, Size(63, 47), INTER_NEAREST, BORDER_REPLICATE, Scalar(), consumer);
    consumer.wait();
    const Mat changed = randomImage(host.size(), host.type(), 77002);
    src.upload(changed);
    device::resize(src, small, Size(63, 47), 0, 0, INTER_AREA, producer);
    consumer.waitFor(producer);
    device::cvtColor(small, gray, COLOR_BGR2GRAY, consumer);
    device::warpAffine(gray, output, identity, Size(63, 47), INTER_NEAREST, BORDER_REPLICATE, Scalar(), consumer);
    consumer.wait();
    EXPECT_EQ(0, cv::norm(output.download(), cpuCvt(oracle(changed, Size(63, 47), 0, 0, INTER_AREA), COLOR_BGR2GRAY), NORM_INF));
}

}}
