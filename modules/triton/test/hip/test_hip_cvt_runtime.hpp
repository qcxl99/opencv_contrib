// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright 2026, The FlagOS Contributors.
#ifndef OPENCV_TRITON_TEST_HIP_CVT_RUNTIME_HPP
#define OPENCV_TRITON_TEST_HIP_CVT_RUNTIME_HPP
// Included after test_hip.cpp's local image/comparison helpers. No extra test
// framework is needed; the caller owns the suite name and the primary stream.
static void cvtRuntimeChecks(hipStream_t stream)
{
    namespace detail = cv::triton::detail;
    const unsigned char fill = 0xA5;
    const cv::Size size(131, 97);
    const cv::Mat hostA = randomImage(size, CV_8UC3, 70101);
    const cv::Mat hostB = randomImage(size, CV_8UC3, 70102);
    DevImg sourceA, sourceB, outputA, outputB, scratch;
    alloc(sourceA, size.height, size.width, CV_8UC3, 2, 0);
    alloc(sourceB, size.height, size.width, CV_8UC3, 3, 0);
    alloc(outputA, size.height, size.width, CV_8UC3, 3, fill);
    alloc(outputB, size.height, size.width, CV_8UC3, 4, fill);
    alloc(scratch, 23, 31, CV_8UC3, 2, fill);
    upload(sourceA, hostA);
    upload(sourceB, hostB);
    hipStream_t other = nullptr;
    HIPOK(hipStreamCreate(&other));

    struct RestoreCvtRuntime
    {
        hipStream_t other;
        hipGraph_t graph = nullptr;
        hipGraphExec_t executable = nullptr;
        ~RestoreCvtRuntime()
        {
            (void)hipDeviceSynchronize();
            if (executable)
                (void)hipGraphExecDestroy(executable);
            if (graph)
                (void)hipGraphDestroy(graph);
            cv::triton::releaseGraphResources();
            detail::reapCompleted();
            detail::setCacheLimits(64, 256);
            detail::setPoisonNewTables(false);
            (void)hipStreamDestroy(other);
        }
    } restore{other};

    const auto exact = [](double)
    {
        return 0.0;
    };
    auto outputMatches = [&](const DevImg &image, const cv::Mat &host, int code)
    {
        cv::Mat reference;
        cv::cvtColor(host, reference, code);
        const cv::Mat parent = downloadParent(image);
        std::string why;
        return compare(roiOf(parent, image), reference, exact, why) &&
               untouchedOutside(parent, image, fill);
    };

    detail::setPoisonNewTables(true);
    for (int code : {cv::COLOR_BGR2HSV, cv::COLOR_BGR2Lab, cv::COLOR_BGR2Luv,
                     cv::COLOR_Lab2BGR, cv::COLOR_Luv2BGR, cv::COLOR_Luv2RGB,
                     cv::COLOR_Luv2LBGR, cv::COLOR_Luv2LRGB})
    {
        // The runtime clamps a zero limit to one; replace that last table
        // with an unrelated resize plan before requesting a fresh color family.
        detail::setCacheLimits(1, 1);
        tr::resize(sourceB.view(), scratch.view(), {31, 23}, cv::INTER_NEAREST, 0,
                   reinterpret_cast<std::uintptr_t>(stream));
        HIPOK(hipStreamSynchronize(stream));
        detail::setCacheLimits(64, 256);
        const auto before = detail::runtimeStats();
        tr::cvtColor(sourceA.view(), outputA.view(), code, 0,
                     reinterpret_cast<std::uintptr_t>(stream));
        tr::cvtColor(sourceB.view(), outputB.view(), code, 0,
                     reinterpret_cast<std::uintptr_t>(other));
        HIPOK(hipStreamSynchronize(stream));
        HIPOK(hipStreamSynchronize(other));
        const auto after = detail::runtimeStats();
        // Evaluate numerical/guard checks even when a statistics assertion fails.
        const bool matchesA = outputMatches(outputA, hostA, code);
        const bool matchesB = outputMatches(outputB, hostB, code);
        std::printf("  cvtColor code=%d uploads=%llu->%llu waits=%llu->%llu exact+guards=%d/%d\n",
                    code, static_cast<unsigned long long>(before.table_uploads),
                    static_cast<unsigned long long>(after.table_uploads),
                    static_cast<unsigned long long>(before.cross_stream_waits),
                    static_cast<unsigned long long>(after.cross_stream_waits), matchesA, matchesB);
        report(
            after.table_uploads > before.table_uploads &&
                after.cross_stream_waits > before.cross_stream_waits &&
                matchesA && matchesB,
            cv::format("cvtColor code=%d poisoned tables: two streams exact and ROI guards intact",
                       code));
    }

    for (int captureCode : {cv::COLOR_BGR2Luv, cv::COLOR_Luv2BGR, cv::COLOR_Luv2RGB,
                            cv::COLOR_Luv2LBGR, cv::COLOR_Luv2LRGB})
    {
        // Warm complete table closures, including the private inverse-Luv layout.
        detail::setCacheLimits(64, 256);
        tr::cvtColor(sourceA.view(), outputA.view(), captureCode, 0,
                     reinterpret_cast<std::uintptr_t>(stream));
        tr::cvtColor(sourceA.view(), outputB.view(), cv::COLOR_BGR2HSV, 0,
                     reinterpret_cast<std::uintptr_t>(stream));
        HIPOK(hipStreamSynchronize(stream));
        const auto beforeCapture = detail::runtimeStats();
        HIPOK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
        tr::cvtColor(sourceA.view(), outputA.view(), captureCode, 0,
                     reinterpret_cast<std::uintptr_t>(stream));
        tr::cvtColor(sourceA.view(), outputB.view(), cv::COLOR_BGR2HSV, 0,
                     reinterpret_cast<std::uintptr_t>(stream));
        HIPOK(hipStreamEndCapture(stream, &restore.graph));
        HIPOK(hipGraphInstantiate(&restore.executable, restore.graph, nullptr, nullptr, 0));
        const auto captured = detail::runtimeStats();

        // A zero cache limit is clamped to one entry by the runtime. Force a
        // different resize module/table through that last slot too, so BOTH
        // captured conversions are genuinely evicted rather than merely shrunk.
        detail::setCacheLimits(1, 1);
        tr::resize(sourceB.view(), scratch.view(), {31, 23}, cv::INTER_NEAREST, 0,
                   reinterpret_cast<std::uintptr_t>(stream));
        HIPOK(hipStreamSynchronize(stream));
        const auto evicted = detail::runtimeStats();
        report(
            captured.capture_launches >= beforeCapture.capture_launches + 2 &&
                captured.capture_pin_holds > beforeCapture.capture_pin_holds &&
                evicted.module_evictions >= captured.module_evictions + 2 &&
                evicted.table_evictions >= captured.table_evictions + 3,
            "cvtColor captured Luv/HSV modules and table closures pinned through actual LRU eviction");

        const cv::Mat changed = randomImage(size, CV_8UC3, 70103 + captureCode);
        upload(sourceA, changed);
        HIPOK(hipMemset(outputA.base, fill, outputA.bytes));
        HIPOK(hipMemset(outputB.base, fill, outputB.bytes));
        HIPOK(hipGraphLaunch(restore.executable, stream));
        HIPOK(hipStreamSynchronize(stream));
        report(
            outputMatches(outputA, changed, captureCode),
            cv::format("captured code=%d replays changed input after LUT/module eviction, ROI guards intact", captureCode));
        report(
            outputMatches(outputB, changed, cv::COLOR_BGR2HSV),
            "captured BGR2HSV replays changed input exactly after table/module eviction, independent dst guards intact");

        HIPOK(hipGraphExecDestroy(restore.executable));
        restore.executable = nullptr;
        HIPOK(hipGraphDestroy(restore.graph));
        restore.graph = nullptr;
        cv::triton::releaseGraphResources();
        detail::reapCompleted();
        report(detail::runtimeStats().capture_pin_holds == 0,
               "public releaseGraphResources drops all cvtColor pins after graph destruction");
    }
}
#endif // OPENCV_TRITON_TEST_HIP_CVT_RUNTIME_HPP
