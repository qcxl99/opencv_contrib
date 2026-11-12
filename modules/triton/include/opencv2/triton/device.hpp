// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Device-pointer entries of the triton module, for builds without cv::cuda.
//
// On a HIP build (TRITON_BACKEND=hip, e.g. Hygon BW1000 / gfx936) OpenCV has no CUDA, so
// cv::cuda::GpuMat exists only as a type whose methods throw, and cv::cuda::Stream carries no
// device stream. The GpuMat entries in opencv2/triton.hpp therefore cannot run there and
// raise StsNotImplemented. These functions are the same three operations with the device
// buffers and the stream named directly: the caller owns the allocations (a framework tensor,
// hipMalloc, ...) and the stream; nothing here allocates, synchronises or falls back to CPU.
//
// They validate exactly what the GpuMat entries validate and go through the same runtime,
// so a result is the one the contract defines (CPU-aligned; see the module description).
// Including this header needs no CUDA or HIP toolkit header.
#ifndef OPENCV_TRITON_DEVICE_HPP
#define OPENCV_TRITON_DEVICE_HPP

#include <cstddef>
#include <cstdint>

#include "opencv2/core.hpp"

namespace cv { namespace triton {

//! @addtogroup triton
//! @{

/** @brief One device-resident 2-D image: the address of its first element, its size, type and
 * row pitch in BYTES. A ROI view is described by its own first element and the parent's pitch. */
struct CV_EXPORTS DeviceImage
{
    std::uintptr_t ptr = 0;   //!< device address of element (0, 0)
    int rows = 0, cols = 0;
    int type = 0;             //!< OpenCV type, e.g. CV_8UC3
    std::size_t step = 0;     //!< row pitch in bytes
};

/** @brief Is a usable device present for `device` on the backend this module was built for?
 * Pure query: no allocation, no submission, no compilation. */
CV_EXPORTS bool haveDevice(int device);

/** @brief cv::triton::resize on device buffers. `dst` must already be sized to dsize; dsize
 * must be explicit. `stream` is the backend's stream handle (hipStream_t / CUstream), 0 for
 * the default stream. Asynchronous: the call returns once the work is submitted. */
CV_EXPORTS void resizeDevice(const DeviceImage& src, const DeviceImage& dst, Size dsize,
                             int interpolation, int device, std::uintptr_t stream);

/** @brief cv::triton::cvtColor on device buffers. dst's channel count is the code's. */
CV_EXPORTS void cvtColorDevice(const DeviceImage& src, const DeviceImage& dst, int code,
                               int device, std::uintptr_t stream);

/** @brief cv::triton::warpAffine on device buffers. `M` is the 2x3 matrix, row-major, as
 * cv::warpAffine takes it; `flags`, `borderMode` and `borderValue` mean what they mean there. */
CV_EXPORTS void warpAffineDevice(const DeviceImage& src, const DeviceImage& dst, const double M[6],
                                 int flags, int borderMode, const Scalar& borderValue,
                                 int device, std::uintptr_t stream);

//! @}

}}  // namespace cv::triton

#endif  // OPENCV_TRITON_DEVICE_HPP
