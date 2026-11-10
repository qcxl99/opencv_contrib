// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_TRITON_PERF_PRECOMP_HPP
#define OPENCV_TRITON_PERF_PRECOMP_HPP
#include "opencv2/ts.hpp"
#include "opencv2/ts/cuda_perf.hpp"
#include "opencv2/triton.hpp"

// cv::cuda::resize is only a labelled timing baseline, and cudawarping is an OPTIONAL dependency
// of perf/ (see CMakeLists.txt). Without this guard a build that has triton but not cudawarping
// -- perfectly legal, since the implementation never uses it -- failed to compile the perf
// module instead of simply skipping the baseline test.
#ifdef OPENCV_TRITON_PERF_HAVE_CUDAWARPING
#include "opencv2/cudawarping.hpp"
#endif
#endif
