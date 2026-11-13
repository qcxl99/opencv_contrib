// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_TRITON_VALIDATION_NUMERICS_HPP
#define OPENCV_TRITON_VALIDATION_NUMERICS_HPP
#include <algorithm>
#include <cmath>
namespace triton_validation { namespace numeric {
inline double resizeF32Atol(int interp) {
    return interp == 0 ? 0.0 : (interp == 1 ? 4.948e-7 : (interp == 2 ? 1.401e-6 : 8.419e-7));
}
inline double warpF32Tolerance(double value) {
    return std::min(5.0e-5*std::max(std::abs(value),1.0),1.0e-4);
}
inline double cvtF32Tolerance(double reference,double inputScale, bool absoluteCap = false) {
    const double scaled = 8*std::ldexp(1.0,-24)*std::max(1.0,std::max(std::abs(reference),inputScale));
    return absoluteCap ? std::min(scaled, 1.0e-4) : scaled;
}
}}
#endif
