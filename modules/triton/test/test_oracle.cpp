// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Compile the reference and private validation helpers independently of the library.
// Separate float32 operations are required (-ffp-contract=off in CMakeLists.txt).
#include "../src/runtime/provider.cpp"
#include "../src/runtime/compile_protocol.cpp"
