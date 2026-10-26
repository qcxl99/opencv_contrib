# PROPOSED optional module. Disabled unless WITH_TRITON=ON, so an ordinary
# OpenCV build needs no Python, Triton or PyTorch.
if(NOT WITH_TRITON)
  ocv_module_disable(triton)
endif()
if(IOS OR WINRT OR (NOT HAVE_CUDA AND NOT TRITON_BACKEND_CUDA))
  ocv_module_disable(triton)
endif()

set(the_description "Triton-backed operations")

# Production dependencies only. cudawarping and ts are TEST-only: the native
# resize is a comparison reference, not a dependency of this implementation.
# imgproc is a PRODUCTION dependency: the interpolation enums (INTER_LINEAR,
# INTER_NEAREST) live there. cudawarping is TEST/PERF-only -- the native resize
# is a comparison reference, not something this implementation calls.
ocv_define_module(triton opencv_core opencv_imgproc
                  OPTIONAL opencv_cudawarping
                  WRAP python)

if(NOT TRITON_BACKEND_CUDA)
  message(STATUS "triton: no backend selected; cv::triton::resize will raise StsNotImplemented")
  return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/CompileTriton.cmake")

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
# cuda_stream_accessor.hpp includes it. Resolve the include dir from whichever
# variable this OpenCV/CMake combination populated rather than assuming one.
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
