// Module-private backend protocol.
//
// Deliberately minimal: a small image view, an execution context and a resize
// request. This is NOT a public cross-platform Mat and NOT a general plugin
// registry -- one backend selected at build time is called by a direct
// function call. A registry would be speculative generality until a second
// platform actually exists.

#ifndef OPENCV_TRITON_BACKEND_HPP
#define OPENCV_TRITON_BACKEND_HPP

#include <cstddef>
#include <cstdint>

#include "opencv2/core.hpp"

namespace cv { namespace triton { namespace backend {

/** Device image description. Carries byte pitch, not element stride, because
 *  that is what the device pointer arithmetic needs. whole_* records the
 *  parent allocation so a ROI view addresses correctly rather than being
 *  mistaken for a full, densely packed buffer. */
struct ImageView
{
    std::uintptr_t ptr = 0;
    int rows = 0;
    int cols = 0;
    int type = 0;            //!< OpenCV type, e.g. CV_8UC3
    std::size_t step = 0;    //!< row pitch IN BYTES
    int whole_rows = 0;      //!< parent allocation height
    int whole_cols = 0;      //!< parent allocation width
    int offset_y = 0;        //!< ROI offset within the parent
    int offset_x = 0;

    bool isRoi() const
    {
        return offset_x != 0 || offset_y != 0 ||
               whole_rows != rows || whole_cols != cols;
    }
    int channels() const { return CV_MAT_CN(type); }
    int depth() const { return CV_MAT_DEPTH(type); }
};

/** Execution context. Device and context must match the one that owns the
 *  buffers; the backend never creates a context of its own. */
struct Context
{
    int device = -1;
    std::uintptr_t stream = 0;   //!< platform stream handle, may be 0 (default)
    const char* backend_name = "";
};

struct ResizeRequest
{
    ImageView src;
    ImageView dst;
    Size dsize;
    double fx = 0;
    double fy = 0;
    int interpolation = 0;
};

/** Outcome of a launch attempt. `variant` names the artifact actually used so
 *  a result can be traced to one compiled specialization. */
struct LaunchResult
{
    bool submitted = false;
    int error_code = 0;
    const char* message = "";
    char variant[64] = {0};
};

/** Is a backend usable on this device right now? */
bool available(int device);

/** Backend name, or "" when unavailable. */
const char* name();

/** Submit the resize. Must not synchronise and must not allocate. */
LaunchResult launchResize(const Context& ctx, const ResizeRequest& req);

}}}  // namespace cv::triton::backend

#endif  // OPENCV_TRITON_BACKEND_HPP
