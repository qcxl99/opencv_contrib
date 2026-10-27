#ifndef OPENCV_TRITON_PRECOMP_HPP
#define OPENCV_TRITON_PRECOMP_HPP

#include "opencv2/triton/core.hpp"
#include "opencv2/core/private.hpp"
#include "opencv2/imgproc.hpp"   // INTER_LINEAR / INTER_NEAREST -- needed by
                                 // core_resize.cpp AND both backends, not
                                 // just the CUDA adapter.

// triton.hpp (the GpuMat-based public API) and cuda_stream_accessor.hpp (which
// unconditionally includes cuda_runtime.h) are CUDA-adapter-only. This
// precompiled header must not pull either in on a build with no CUDA toolkit
// (e.g. BW1000/HIP): only cuda_resize_adapter.cpp is compiled against them,
// and only as part of the CUDA-backend source set (see CMakeLists.txt).
#ifdef OPENCV_TRITON_HAVE_CUDA_ADAPTER
#include "opencv2/triton.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#endif

#include "backend/backend.hpp"

#endif
