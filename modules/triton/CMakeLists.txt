# PROPOSED optional module. Disabled unless WITH_TRITON=ON, so an ordinary
# OpenCV build needs no Python, Triton or PyTorch.
if(NOT WITH_TRITON)
  ocv_module_disable(triton)
endif()
# The platform-agnostic core (opencv2/triton/core.hpp, resizeCore()) needs
# neither CUDA nor HIP, but this module is pointless with no backend to run
# resizeCore() against, so the module as a whole still requires picking one.
if(IOS OR WINRT OR (NOT HAVE_CUDA AND NOT TRITON_BACKEND_CUDA AND NOT TRITON_BACKEND_HIP))
  ocv_module_disable(triton)
endif()

set(the_description "Triton-backed operations")

# Mutually exclusive by construction: CUDA wins if somehow both were set.
if(TRITON_BACKEND_CUDA)
  set(TRITON_BACKEND_HIP OFF)
endif()

# --- explicit source list, NOT ocv_define_module's glob ---------------------
# ocv_define_module()'s default ocv_glob_module_sources() recursively globs
# ALL of src/*.cpp, including both backend subdirectories and BOTH adapter
# files (cuda_resize_adapter.cpp and gpumat_adapter_stub.cpp). Those two
# adapters export the same three symbols and must never both be linked into
# one binary, so this module assembles its source list by hand instead of
# relying on the glob, then calls the same ocv_add_module -> ocv_set_module_
# sources -> ocv_module_include_directories -> ocv_create_module sequence
# that ocv_define_module() would have run.
set(_triton_srcs "${CMAKE_CURRENT_LIST_DIR}/src/core_resize.cpp")

if(TRITON_BACKEND_CUDA)
  list(APPEND _triton_srcs
       "${CMAKE_CURRENT_LIST_DIR}/src/cuda_resize_adapter.cpp"
       "${CMAKE_CURRENT_LIST_DIR}/src/backend/cuda/resize.cpp")
elseif(TRITON_BACKEND_HIP)
  list(APPEND _triton_srcs
       "${CMAKE_CURRENT_LIST_DIR}/src/gpumat_adapter_stub.cpp"
       "${CMAKE_CURRENT_LIST_DIR}/src/backend/hip/resize.cpp")
else()
  message(STATUS "triton: no backend selected; cv::triton::resize will raise "
                 "StsNotImplemented, and resizeCore() will raise too")
  list(APPEND _triton_srcs "${CMAKE_CURRENT_LIST_DIR}/src/gpumat_adapter_stub.cpp")
endif()

file(GLOB _triton_hdrs
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/triton.hpp"
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/triton/*.hpp")
file(GLOB _triton_int_hdrs
     "${CMAKE_CURRENT_LIST_DIR}/src/*.hpp"
     "${CMAKE_CURRENT_LIST_DIR}/src/backend/*.hpp")

# Production dependencies only. cudawarping and ts are TEST-only: the native
# resize is a comparison reference, not a dependency of this implementation.
# imgproc is a PRODUCTION dependency: the interpolation enums (INTER_LINEAR,
# INTER_NEAREST) live there. cudawarping is TEST/PERF-only and only relevant
# to the CUDA adapter/tests, never to the HIP path.
ocv_add_module(triton opencv_core opencv_imgproc
               OPTIONAL opencv_cudawarping
               WRAP python)
ocv_set_module_sources(HEADERS ${_triton_hdrs}
                       SOURCES ${_triton_srcs} ${_triton_int_hdrs})
ocv_module_include_directories()
ocv_create_module()

ocv_add_accuracy_tests()
ocv_add_perf_tests()
ocv_add_samples()

if(NOT TRITON_BACKEND_CUDA AND NOT TRITON_BACKEND_HIP)
  return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/CompileTriton.cmake")

if(TRITON_BACKEND_CUDA)

  ocv_target_compile_definitions(${the_module} PRIVATE OPENCV_TRITON_HAVE_CUDA_ADAPTER=1)

  set(TRITON_ARTIFACT_DIR "${CMAKE_BINARY_DIR}/modules/triton/generated/cuda")
  compile_triton_resize(
    KERNEL      "${CMAKE_CURRENT_LIST_DIR}/kernels/resize.py"
    TOOL        "${CMAKE_CURRENT_LIST_DIR}/tools/compile_resize.py"
    OUTPUT_DIR  "${TRITON_ARTIFACT_DIR}"
    MANIFEST    "${CMAKE_BINARY_DIR}/modules/triton/generated/kernels_manifest.json"
  )

  ocv_target_compile_definitions(${the_module} PRIVATE
    HAVE_TRITON_BACKEND=1
    OPENCV_TRITON_ARTIFACT_DIR="${TRITON_ARTIFACT_DIR}")

  # Headers: cuda.h for the driver API, and cuda_runtime.h because OpenCV's
  # cuda_stream_accessor.hpp includes it. Resolve the include dir from
  # whichever variable this OpenCV/CMake combination populated rather than
  # assuming one.
  set(_triton_cuda_inc "")
  foreach(_v CUDAToolkit_INCLUDE_DIRS CUDA_INCLUDE_DIRS CUDA_TOOLKIT_INCLUDE)
    if(${_v})
      list(APPEND _triton_cuda_inc ${${_v}})
    endif()
  endforeach()
  if(NOT _triton_cuda_inc AND CUDAToolkit_ROOT)
    list(APPEND _triton_cuda_inc "${CUDAToolkit_ROOT}/include")
  endif()
  if(NOT _triton_cuda_inc)
    message(FATAL_ERROR "triton: could not resolve the CUDA include directory; "
                        "pass -DCUDAToolkit_INCLUDE_DIRS=<path>")
  endif()
  message(STATUS "triton: CUDA includes ${_triton_cuda_inc}")
  ocv_target_include_directories(${the_module} PRIVATE ${_triton_cuda_inc})

  # The backend speaks to the CUDA driver API directly; it does not need the
  # runtime API or NPP.
  set(_triton_cuda_lib "")
  if(CUDA_CUDA_LIBRARY)
    set(_triton_cuda_lib ${CUDA_CUDA_LIBRARY})
  elseif(TARGET CUDA::cuda_driver)
    set(_triton_cuda_lib CUDA::cuda_driver)
  else()
    set(_triton_cuda_lib cuda)
  endif()
  ocv_target_link_libraries(${the_module} LINK_PRIVATE ${_triton_cuda_lib})

  add_dependencies(${the_module} opencv_triton_kernels)

elseif(TRITON_BACKEND_HIP)

  # No cv::cuda::GpuMat/cv::cuda::Stream anywhere on this path: the stub
  # (added to _triton_srcs above) already provides the GpuMat-shaped resize()
  # as a permanent StsNotImplemented. This build does not require
  # opencv_cudawarping or any CUDA toolkit; callers reach the module through
  # resizeCore() directly.
  set(TRITON_ARTIFACT_DIR "${CMAKE_BINARY_DIR}/modules/triton/generated/hip")
  compile_triton_resize(
    KERNEL      "${CMAKE_CURRENT_LIST_DIR}/kernels/resize_hip.py"
    TOOL        "${CMAKE_CURRENT_LIST_DIR}/tools/compile_resize_hip.py"
    OUTPUT_DIR  "${TRITON_ARTIFACT_DIR}"
    MANIFEST    "${CMAKE_BINARY_DIR}/modules/triton/generated/kernels_manifest_hip.json"
  )

  ocv_target_compile_definitions(${the_module} PRIVATE
    HAVE_TRITON_BACKEND=1
    OPENCV_TRITON_ARTIFACT_DIR="${TRITON_ARTIFACT_DIR}"
    # hip/hip_common.h only auto-defines __HIP_PLATFORM_AMD__ for
    # clang+__HIP__ (i.e. hipcc compiling a recognized HIP source file).
    # This module is compiled by the ordinary C++ compiler (g++/clang++ as
    # OpenCV's toolchain, not hipcc) against plain .cpp files, so the DTK
    # header's own comment applies: "Other compiler (GCC,ICC,etc) need to
    # set one of these macros explicitly."
    __HIP_PLATFORM_AMD__=1)

  # HIP toolkit location: DTK on this platform (ROCM_PATH/HIP_PATH both point
  # at /opt/dtk), not a system ROCm install -- resolve via whichever the
  # caller's environment set, rather than assuming a path.
  if(NOT TRITON_HIP_ROOT)
    if(DEFINED ENV{HIP_PATH})
      set(TRITON_HIP_ROOT "$ENV{HIP_PATH}")
    elseif(DEFINED ENV{ROCM_PATH})
      set(TRITON_HIP_ROOT "$ENV{ROCM_PATH}")
    endif()
  endif()
  if(NOT TRITON_HIP_ROOT OR NOT EXISTS "${TRITON_HIP_ROOT}/include/hip/hip_runtime.h")
    message(FATAL_ERROR "triton: could not resolve the HIP include directory; "
                        "pass -DTRITON_HIP_ROOT=<path to DTK/ROCm> "
                        "(expects <path>/include/hip/hip_runtime.h)")
  endif()
  message(STATUS "triton: HIP includes ${TRITON_HIP_ROOT}/include")
  ocv_target_include_directories(${the_module} PRIVATE "${TRITON_HIP_ROOT}/include")

  # amdhip64 provides both the HIP runtime and driver-style Module/Launch
  # entry points used here (hipModuleLoad, hipModuleLaunchKernel, ...).
  find_library(TRITON_HIP_LIBRARY amdhip64
               PATHS "${TRITON_HIP_ROOT}/lib" "${TRITON_HIP_ROOT}/../lib"
               NO_DEFAULT_PATH)
  if(NOT TRITON_HIP_LIBRARY)
    message(FATAL_ERROR "triton: could not find libamdhip64 under ${TRITON_HIP_ROOT}; "
                        "pass -DTRITON_HIP_LIBRARY=<path to libamdhip64.so>")
  endif()
  message(STATUS "triton: HIP library ${TRITON_HIP_LIBRARY}")
  ocv_target_link_libraries(${the_module} LINK_PRIVATE ${TRITON_HIP_LIBRARY})

  add_dependencies(${the_module} opencv_triton_kernels)

endif()
