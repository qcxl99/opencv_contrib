// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Stub for the GpuMat-based public API when no CUDA adapter is compiled in --
// i.e. every build EXCEPT TRITON_BACKEND_CUDA (HIP/BW1000 builds included).
// Compiled INSTEAD of cuda_resize_adapter.cpp, never alongside it (see
// CMakeLists.txt): the two provide the same three symbols
// (haveBackend/backendName/resize) and must never both be linked in.
//
// This file does NOT include cuda_stream_accessor.hpp (so it carries no
// CUDA-runtime-header dependency), but it does include opencv2/triton.hpp,
// which pulls in opencv2/core/cuda.hpp for the cuda::GpuMat/cuda::Stream
// *types* used in resize()'s signature. That header has no HAVE_CUDA guard
// and requires no CUDA toolkit -- cv::cuda::GpuMat exists as a type on every
// OpenCV build; only cv::cuda::GpuMat's own methods throw_no_cuda() when
// WITH_CUDA was off. A BW1000/HIP build of this module therefore still
// builds and exports a GpuMat-shaped cv::triton::resize -- it simply always
// raises, same as the original no-backend stub did.

#include "opencv2/triton.hpp"
#include "opencv2/core/private.hpp"

namespace cv { namespace triton {

bool haveBackend()
{
    return false;
}

String backendName()
{
    return String();
}

void resize(InputArray, OutputArray, Size, double, double, int, cuda::Stream&)
{
    CV_Error(Error::StsNotImplemented,
             "cv::triton::resize (GpuMat overload) requires the module to be built with "
             "TRITON_BACKEND_CUDA. This platform/build uses cv::triton::resizeCore() "
             "(opencv2/triton/core.hpp) directly instead.");
}

}}  // namespace cv::triton
