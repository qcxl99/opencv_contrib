// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_TRITON_PERF_PRECOMP_HPP
#define OPENCV_TRITON_PERF_PRECOMP_HPP
#include "opencv2/ts.hpp"
#include "opencv2/ts/ts_perf.hpp"
#include "../validation/device.hpp"
#if defined(OPENCV_TRITON_PERF_HAVE_CUDAWARPING) || defined(OPENCV_TRITON_PERF_HAVE_CUDAIMGPROC)
#include "opencv2/ts/cuda_perf.hpp"
#endif
#include "opencv2/imgproc.hpp"

// cv::cuda::resize is only a labelled timing baseline, and cudawarping is an OPTIONAL dependency
// of perf/ (see CMakeLists.txt). Without this guard a build that has triton but not cudawarping
// -- perfectly legal, since the implementation never uses it -- failed to compile the perf
// module instead of simply skipping the baseline test.
#ifdef OPENCV_TRITON_PERF_HAVE_CUDAWARPING
#include "opencv2/cudawarping.hpp"
#endif
// cv::cuda::cvtColor lives in cudaimgproc, a separate optional dependency of perf/ with its
// own guard: a build may have either, both or neither.
#ifdef OPENCV_TRITON_PERF_HAVE_CUDAIMGPROC
#include "opencv2/cudaimgproc.hpp"
#endif
#endif
