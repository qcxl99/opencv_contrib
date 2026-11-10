// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Module-private runtime JIT (P1; kernelforge-lab BACKLOG B5). Not installed.
//
// The one entry the adapters call is runtime::resize(). It
//   * normalises and validates the request (provider.hpp normalize(): CPU cv::resize
//     parameter resolution, int32 ranges, overlap), and serves the P1 routes only (copy,
//     nearest, linear, area_fast); cubic and fractional area are Unsupported/route_not_enabled;
//   * finds the binary for the request's compile key: in-process module cache -> disk cache
//     (verified: compile_protocol.hpp verify_artifact) -> the Torch-free compile helper in a
//     separate process (compile_key), never Python in this process;
//   * builds the coordinate/coefficient tables in C++ (reference_cpu.hpp, the same code as
//     the portable oracle), uploads them on the caller's stream and records an event; a call on
//     another stream waits on that event;
//   * launches with the manifest's parameter list. The two trailing scratch pointers are passed
//     as null only when launch_abi_ok() holds: a validated compiler ABI, zero scratch bytes,
//     and exactly (global_scratch, profile_scratch) after the runtime arguments;
//   * keeps every module and table a launch uses alive until that launch's completion event:
//     cache eviction drops the cache's reference only.
//
// Costs to know, measured on the 12.8 driver: cuModuleLoadData, cuModuleUnload and cuMemFree block
// until the context's in-flight work has drained. So:
//   * the first use of a key in a context stalls the calling thread on its own earlier work;
//   * nothing that can unload is destroyed under the state mutex;
//   * tables are stream-ordered memory (cuMemAllocAsync / cuMemFreeAsync), so evicting them, the
//     common case once many shapes are in use, does not stall the application's thread.
//
// Linux/CUDA only in this milestone.
#ifndef OPENCV_TRITON_RUNTIME_HPP
#define OPENCV_TRITON_RUNTIME_HPP

#include <cstdint>
#include <string>

#include "opencv2/core.hpp"
#include "../core.hpp"   // module-internal; src/core.hpp is not on the include path
#include "provider.hpp"

namespace cv { namespace triton { namespace runtime {

/** Is the CUDA driver usable for `device`? Pure query: no allocation, no submission. */
bool deviceAvailable(int device);
const char* name();   // "cuda-jit"

/** Run one resize. `req` views must belong to the context current on the calling thread.
 *  Returns Submitted (work is on ex.stream), Unsupported (nothing submitted, dst untouched)
 *  or Error. `route` (if given) receives the route name. */
kf_resize_rt::Status resize(const kf_resize_rt::ResizeRequest& req, const kf_resize_rt::ExecContext& ex,
                            std::string* route = nullptr);

}}}  // namespace cv::triton::runtime

namespace cv { namespace triton { namespace detail {

/** ImageView -> the runtime's device view (alloc_base/bytes = the owning allocation, 0 = unknown). */
kf_resize_rt::DeviceView toDeviceView(const ImageView& v, int device, std::uintptr_t alloc_base, std::size_t alloc_bytes);
/** The common entry of resizeCore() and the GpuMat adapter. dsize empty = fx/fy given (the
 *  caller's values, passed through unchanged). Raises the mapped OpenCV error. */
void resizeDevice(const kf_resize_rt::DeviceView& src, const kf_resize_rt::DeviceView& dst, Size dsize,
                  double fx, double fy, int interpolation, int device, std::uintptr_t stream);

}}}  // namespace cv::triton::detail

namespace cv { namespace triton { namespace detail {

/** Test-only introspection and control (test/test_runtime.cpp). Exported because OpenCV
 *  builds with hidden visibility; declared only in this private header. */
struct RuntimeStats
{
    long long helper_compiles = 0;     //!< helper responses "compiled"
    long long helper_cached = 0;       //!< helper responses "cached" (another process built it)
    long long disk_hits = 0;           //!< verified artifact loaded without running the helper
    long long disk_rejected = 0;       //!< artifact on disk that failed verification
    long long module_loads = 0, module_hits = 0, module_evictions = 0, module_frees = 0;
    long long table_uploads = 0, table_hits = 0, table_evictions = 0, table_frees = 0;
    long long deferred = 0;            //!< evictions of a resource still held by an in-flight launch
    long long cross_stream_waits = 0;  //!< launches that waited on another stream's table upload
    long long launches = 0, copies = 0;
    long long pending = 0;             //!< launches whose completion has not been observed yet
    std::string last_route, last_key, last_artifact;
};
CV_EXPORTS RuntimeStats runtimeStats();
CV_EXPORTS void resetRuntimeStats();
/** Cache capacities in entries (defaults 64 modules, 256 table sets). Shrinking evicts now. */
CV_EXPORTS void setCacheLimits(int modules, int tables);
/** Observe completed launches now (normally done at every call). */
CV_EXPORTS void reapCompleted();
/** Fill newly allocated table buffers with 0xFF before the upload, so a launch that read a
 *  table before its upload completed produces garbage instead of plausible numbers. */
CV_EXPORTS void setPoisonNewTables(bool on);
/** Resolved paths: "resources=<dir>;kernels=<file>;helper=<file>;cache=<dir>;python=<exe>;compiler=<id>". */
CV_EXPORTS std::string runtimeConfig();

}}}  // namespace cv::triton::detail

#endif
