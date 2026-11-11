// Minimal buildable example: upload -> cv::triton::cvtColor -> download.
//
// The conversion codes this module serves are listed in opencv2/triton.hpp. Anything outside
// that list (HSV2BGR, HLS, CV_32F) raises cv::Error::StsUnsupportedFormat rather than falling
// back, so an unsupported request is a visible error, not a silent change of result.
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/triton.hpp>

int main(int argc, char** argv)
{
    if (!cv::triton::haveBackend())
    {
        std::cerr << "No Triton backend for this device.\n";
        return 1;
    }
    std::cout << "backend: " << cv::triton::backendName() << "\n";

    cv::Mat host = (argc > 1) ? cv::imread(argv[1], cv::IMREAD_COLOR)
                              : cv::Mat(720, 1280, CV_8UC3, cv::Scalar(64, 128, 192));
    if (host.empty()) { std::cerr << "failed to read input\n"; return 1; }

    cv::cuda::GpuMat src(host), dst;
    cv::cuda::Stream stream;

    // Asynchronous: the call returns after submitting, and src/dst must stay alive until the
    // work completes. The destination is created if it is empty and reused if it already has
    // the right size and type.
    cv::triton::cvtColor(src, dst, cv::COLOR_BGR2GRAY, stream);
    stream.waitForCompletion();   // wait before reading the result

    cv::Mat out;
    dst.download(out);
    std::cout << "converted " << host.size() << " BGR -> GRAY " << out.size()
              << " (" << out.channels() << " channel)\n";
    if (argc > 2) cv::imwrite(argv[2], out);
    return 0;
}
