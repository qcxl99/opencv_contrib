// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Module-private runtime JIT. Not installed.
// Adapters call resize(), cvtColor() or warpAffine(). Each validates its supported request,
// obtains a verified artifact (context-local module cache -> disk cache -> separate compile
// helper process), prepares stream-ordered tables when needed, and submits on the caller's
// stream. Resize serves copy, nearest, linear, cubic, area_fast and fractional AREA routes.
// Supported input subsets remain operation/backend-specific; unsupported calls do not fall back.
// C++ loads verified cubin/hsaco entry metadata; Python is used only by the compile helper.
// In-flight work retains modules/tables; graph captures pin resources until the caller releases
// them after all dependent graphs and executions are finished. See the public API contract.
// CUDA driver 12.8 observations: module load/unload can wait for prior context work. Releases
// occur outside the runtime lock; tables use stream-ordered allocation/free.
// CUDA and HIP implementations are selected at build time. Their capability and test results
// are backend-specific, not interchangeable. See doc/maintenance.markdown.
#ifndef OPENCV_TRITON_RUNTIME_HPP
#define OPENCV_TRITON_RUNTIME_HPP

#include <cstdint>
#include <string>

#include "opencv2/core.hpp"
#include "../core.hpp"   // module-internal; src/core.hpp is not on the include path
#include "provider.hpp"
#include "cvtcolor.hpp"
#include "warpaffine.hpp"

namespace cv { namespace triton { namespace runtime {

/** Is the selected backend usable for `device`? Pure query: no allocation or submission. */
bool deviceAvailable(int device);
const char* name();   // "cuda-jit" or "hip-jit"

/** Run one resize. `req` views must belong to the context current on the calling thread.
 *  Returns Submitted (work is on ex.stream), Unsupported (nothing submitted, dst untouched)
 *  or Error. `route` (if given) receives the route name. */
kf_resize_rt::Status resize(const kf_resize_rt::ResizeRequest& req, const kf_resize_rt::ExecContext& ex,
                            std::string* route = nullptr);

/** Run one cvtColor. Same contract as resize(): Submitted, Unsupported (nothing submitted,
 *  dst untouched) or Error. `opName` receives the conversion's name. */
kf_resize_rt::Status cvtColor(const kf_cvtcolor::Request& req, const kf_resize_rt::ExecContext& ex,
                              std::string* opName = nullptr);

/** Run one warpAffine. Same contract as resize(): Submitted, Unsupported (nothing submitted,
 *  dst untouched) or Error. */
kf_resize_rt::Status warpAffine(const kf_warpaffine::Request& req, const kf_resize_rt::ExecContext& ex);

}}}  // namespace cv::triton::runtime

namespace cv { namespace triton { namespace detail {

/** ImageView -> the runtime's device view (alloc_base/bytes = the owning allocation, 0 = unknown). */
kf_resize_rt::DeviceView toDeviceView(const ImageView& v, int device, std::uintptr_t alloc_base, std::size_t alloc_bytes);
/** The common entry of resizeCore() and the GpuMat adapter. dsize empty = fx/fy given (the
 *  caller's values, passed through unchanged). Raises the mapped OpenCV error. */
void resizeDevice(const kf_resize_rt::DeviceView& src, const kf_resize_rt::DeviceView& dst, Size dsize,
                  double fx, double fy, int interpolation, int device, std::uintptr_t stream);
/** cvtColor counterpart: validates, calls runtime::cvtColor, raises the mapped OpenCV error. */
void cvtColorDevice(const kf_cvtcolor::Request& req, int device, std::uintptr_t stream);
/** warpAffine counterpart: validates, calls runtime::warpAffine, raises the mapped OpenCV error. */
void warpAffineDevice(const kf_warpaffine::Request& req, int device, std::uintptr_t stream);

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
    long long capture_launches = 0;    //!< launches recorded into a capturing stream
    long long capture_pin_holds = 0;   //!< resources currently pinned for captured graphs
    std::string last_route, last_key, last_artifact;
};
CV_EXPORTS RuntimeStats runtimeStats();
CV_EXPORTS void resetRuntimeStats();
/** Cache capacities in entries (defaults 64 modules, 256 table sets). Shrinking evicts now. */
CV_EXPORTS void setCacheLimits(int modules, int tables);
/** Observe completed launches now (normally done at every call). */
CV_EXPORTS void reapCompleted();
/** Release the resources pinned for captured graphs.
 *
 * A graph recorded from a capturing stream is launched later, possibly after its module and
 * tables would have been evicted, so the runtime pins them at capture time. CUDA does not
 * notify when the caller destroys a graph, so the pin is released here instead: call this once
 * the graphs you captured are destroyed or no longer used. Returning them early and then
 * launching a graph that reads them is use-after-free, which no API can detect for you. */
CV_EXPORTS void releaseCapturePins();
/** Fill newly allocated table buffers with 0xFF before the upload, so a launch that read a
 *  table before its upload completed produces garbage instead of plausible numbers. */
CV_EXPORTS void setPoisonNewTables(bool on);
/** Resolved paths: "resources=<dir>;kernels=<file>;helper=<file>;cache=<dir>;python=<exe>;compiler=<id>". */
CV_EXPORTS std::string runtimeConfig();

}}}  // namespace cv::triton::detail

#endif
