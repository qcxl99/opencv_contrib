// Portable reference for resize (contract resize/0.15.0): OpenCV 5.0.0 CPU cv::resize as
// computed by the pinned reference configuration (contract.yaml semantics.oracle_build:
// resize.cpp's generic code, 128-bit SIMD, no FMA, no IPP, no HAL).
//
// Why it exists (BACKLOG B1): the CPU result is a property of the OpenCV build. An IPP-HAL
// build of the same version differs on 507 of 2069 cases (runs/cpu-build-dependence-0001),
// so a test that compares against the host's own cv::resize tests the host build, not the
// contract. This file computes the contract on any host. It is a C++ port of
// operators/resize/reference_cpu.py, line for line, and is checked bit-exact against it and
// against the pinned cv2 (tests/test_reference_cpu_cpp.py).
//
// It must be compiled with -ffp-contract=off (no fused multiply-add) and on a target whose
// float evaluation is IEEE single (FLT_EVAL_METHOD == 0). The 128-bit SIMD/tail split of the
// reference build is reproduced with explicit constants, independent of the host's SIMD.
//
// Two halves:
//   * plan: size resolution, routing and the coordinate/coefficient tables. The module's
//     C++ runtime builds its device tables from this half (BACKLOG B5).
//   * evaluate: the scalar computation over those tables, the test oracle.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kf_resize_ref {

enum Interp { INTER_NEAREST = 0, INTER_LINEAR = 1, INTER_CUBIC = 2, INTER_AREA = 3 };
enum Route { ROUTE_COPY, ROUTE_NEAREST, ROUTE_LINEAR, ROUTE_CUBIC, ROUTE_AREA_FAST, ROUTE_AREA };

const char* route_name(Route r);

// cv::resize parameter resolution. dsize wins when both sides are > 0; otherwise fx/fy
// (both > 0) give dst = cvRound(src * f) and inverse scales that ARE fx/fy.
struct Size2 { int dst_h, dst_w; double inv_y, inv_x; };
Size2 resolve_size(int src_h, int src_w, int dst_w, int dst_h, double fx, double fy);

struct RouteInfo {
    Route kind;
    bool area_mode = false;   // linear path with AREA's coefficient formula (AREA upscale)
    int ix = 0, iy = 0;       // integer ratio of ROUTE_AREA_FAST
};
RouteInfo route(int src_h, int src_w, int dst_h, int dst_w, double inv_y, double inv_x, int interp);

// One axis of the separable path: source offsets, float32 coefficients [dst x ksize],
// and (horizontal axis) the xmin/xmax border markers.
struct Axis {
    std::vector<int> ofs;
    std::vector<float> coeffs;
    int ksize = 2, xmin = 0, xmax = 0;
};
Axis generic_axis(int dst, int src, double inv_scale, int ksize, bool area_mode, bool horizontal);
// saturate_cast<short>(c * INTER_RESIZE_COEF_SCALE)
std::vector<int> fixed_coeffs(const std::vector<float>& coeffs);

// computeResizeAreaTab, padded to the longest tap list.
struct AreaTab {
    int taps = 0;                 // K
    std::vector<int> src_index;   // [dst x K]
    std::vector<float> alpha;     // [dst x K]
    std::vector<int> count;       // [dst]
};
AreaTab area_tab(int src, int dst, double scale);

// An image in host memory: rows of `cols * channels` elements, `step` bytes apart.
struct ImageRef {
    void* data;
    int rows, cols, channels;
    std::size_t step;
    bool is_f32;
};

// cv::resize on the host. dst.rows/cols must already be the resolved size (resolve_size).
// inv_y / inv_x are the resolved inverse scales. Throws std::invalid_argument on misuse.
void resize(const ImageRef& src, const ImageRef& dst, double inv_y, double inv_x, int interp);

}  // namespace kf_resize_ref

// C ABI for the Python cross-check (ctypes). Returns 0 on success, -1 on a bad argument.
extern "C" int kf_resize_reference(const void* src, int src_rows, int src_cols, std::size_t src_step,
                                   void* dst, int dst_rows, int dst_cols, std::size_t dst_step,
                                   int channels, int is_f32, int interp, double inv_y, double inv_x);
extern "C" int kf_resize_reference_route(int src_h, int src_w, int dst_h, int dst_w,
                                         double inv_y, double inv_x, int interp, int* ix, int* iy,
                                         int* area_mode);
