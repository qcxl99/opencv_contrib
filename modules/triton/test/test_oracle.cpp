// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// The test binary compiles its own copy of the portable CPU oracle and the artifact verifier.
// The reference must not depend on the library under test, nor on the host build's
// cv::resize, whose result depends on the build (IPP, HALs, SIMD width). The target is
// compiled with -ffp-contract=off (CMakeLists.txt), as the oracle requires.
#include "../src/runtime/reference_cpu.cpp"
#include "../src/runtime/provider.cpp"
#include "../src/runtime/compile_protocol.cpp"
