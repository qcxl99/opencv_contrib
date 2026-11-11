# PROPOSED optional module: cv::triton::resize through a runtime JIT of Triton kernels.
#
# Off by default, like any module that needs an external toolchain (precedent: cnn_3dobj):
# enable it with -DBUILD_opencv_triton=ON. An ordinary OpenCV build needs no Python, Triton or
# PyTorch, and even an enabled build never needs PyTorch.
#
# Execution: kernels are compiled on first use by a separate, Torch-free helper process
# (tools/compile_helper.py) into a verified on-disk cache; the call path itself is C++
# (src/runtime/). Nothing is compiled at OpenCV build time.
#
# Scope of this submission: Linux + CUDA (driver API). Windows and a HIP/BW1000 backend are
# not part of it; the module is disabled there rather than silently half-built.
set(BUILD_opencv_triton_INIT OFF)

if(IOS OR WINRT OR WIN32 OR APPLE OR NOT HAVE_CUDA)
  ocv_module_disable(triton)
endif()

set(the_description "Triton-backed operations (runtime JIT)")

# The interpreter the compile helper runs under, and the PYTHONPATH that makes `import triton`
# work for it. Distinct from PYTHON3_EXECUTABLE (the Python bindings' interpreter), though it
# defaults to it. Both can be overridden at run time (OPENCV_TRITON_PYTHON[PATH]).
set(TRITON_PYTHON_EXECUTABLE "${PYTHON3_EXECUTABLE}" CACHE FILEPATH "Python used by the Triton compile helper")
set(TRITON_PYTHONPATH "" CACHE STRING "PYTHONPATH for the Triton compile helper (where `triton` is importable)")

set(_triton_compiler "unknown")
if(TRITON_PYTHON_EXECUTABLE)
  execute_process(COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${TRITON_PYTHONPATH}"
                          "${TRITON_PYTHON_EXECUTABLE}" -c "import sys; sys.modules['torch'] = None; import triton; print(triton.__version__)"
                  OUTPUT_VARIABLE _triton_version OUTPUT_STRIP_TRAILING_WHITESPACE
                  RESULT_VARIABLE _triton_rc ERROR_QUIET)
  if(_triton_rc EQUAL 0 AND _triton_version)
    set(_triton_compiler "triton-${_triton_version}")
  endif()
endif()
if(_triton_compiler STREQUAL "unknown")
  message(WARNING "triton: `import triton` failed under ${TRITON_PYTHON_EXECUTABLE} (PYTHONPATH=${TRITON_PYTHONPATH}); "
                  "the module builds, but every cache miss will report the toolchain as unavailable")
endif()
message(STATUS "triton: compile helper ${TRITON_PYTHON_EXECUTABLE}, ${_triton_compiler}")

set(_triton_srcs
    "${CMAKE_CURRENT_LIST_DIR}/src/core_resize.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/cuda_resize_adapter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/cuda_runtime.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/reference_cpu.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/provider.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/cvtcolor.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/warpaffine.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/compile_protocol.cpp")
# Installed headers: only the public umbrella. src/core.hpp is deliberately NOT here -- it is a
# raw device-pointer layer the plan code shares with the GpuMat adapter, not an API for callers.
file(GLOB _triton_hdrs "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/triton.hpp")
file(GLOB _triton_int_hdrs "${CMAKE_CURRENT_LIST_DIR}/src/*.hpp" "${CMAKE_CURRENT_LIST_DIR}/src/runtime/*.hpp")

# imgproc: the interpolation enums. cudawarping is not a dependency of the implementation; it is
# the performance baseline of perf/ only (a different computation: cv::cuda semantics).
ocv_add_module(triton opencv_core opencv_imgproc WRAP python)
ocv_set_module_sources(HEADERS ${_triton_hdrs} SOURCES ${_triton_srcs} ${_triton_int_hdrs})
ocv_module_include_directories()
ocv_create_module()

# --- CUDA driver API --------------------------------------------------------------------
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
  message(FATAL_ERROR "triton: could not resolve the CUDA include directory; pass -DCUDAToolkit_INCLUDE_DIRS=<path>")
endif()
set(_triton_cuda_lib "")
if(CUDA_CUDA_LIBRARY)
  set(_triton_cuda_lib ${CUDA_CUDA_LIBRARY})
elseif(TARGET CUDA::cuda_driver)
  set(_triton_cuda_lib CUDA::cuda_driver)
else()
  set(_triton_cuda_lib cuda)
endif()
ocv_target_include_directories(${the_module} PRIVATE ${_triton_cuda_inc})
ocv_target_link_libraries(${the_module} LINK_PRIVATE ${_triton_cuda_lib} ${CMAKE_DL_LIBS})
ocv_target_compile_definitions(${the_module} PRIVATE OPENCV_TRITON_HAVE_CUDA_ADAPTER=1)

# The plan tables are float32 arithmetic that must round op by op, as the reference does.
target_compile_options(${the_module} PRIVATE -ffp-contract=off)

# --- resources: kernels source + compile helper -----------------------------------------
# Installed under <share>/triton and found at run time relative to the library, so an install
# tree can be moved; an uninstalled build falls back to this source directory.
set(_triton_res_install "${OPENCV_OTHER_INSTALL_PATH}/triton")
file(RELATIVE_PATH _triton_rel "${CMAKE_INSTALL_PREFIX}/${OPENCV_LIB_INSTALL_PATH}" "${CMAKE_INSTALL_PREFIX}/${_triton_res_install}")
ocv_target_compile_definitions(${the_module} PRIVATE
  OPENCV_TRITON_REL_RESOURCES="${_triton_rel}"
  OPENCV_TRITON_SOURCE_RESOURCES="${CMAKE_CURRENT_LIST_DIR}"
  OPENCV_TRITON_PYTHON_DEFAULT="${TRITON_PYTHON_EXECUTABLE}"
  OPENCV_TRITON_PYTHONPATH_DEFAULT="${TRITON_PYTHONPATH}"
  OPENCV_TRITON_COMPILER_DEFAULT="${_triton_compiler}")
install(FILES "${CMAKE_CURRENT_LIST_DIR}/kernels/resize_cpu.py"
              "${CMAKE_CURRENT_LIST_DIR}/kernels/cvtcolor_cpu.py"
              "${CMAKE_CURRENT_LIST_DIR}/kernels/warpaffine_cpu.py"
        DESTINATION "${_triton_res_install}/kernels" COMPONENT libs)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/tools/compile_helper.py" DESTINATION "${_triton_res_install}/tools" COMPONENT libs)

# --- tests / perf / samples -------------------------------------------------------------
ocv_add_accuracy_tests()
ocv_add_perf_tests()
ocv_add_samples()
# perf/ only: cv::cuda::resize as a labelled timing baseline. The implementation, the accuracy
# tests and the sample never use it, so cudawarping is an OPTIONAL dependency -- and it MUST be
# checked before being linked. Linking it unconditionally meant a build with triton but without
# cudawarping (a legal configuration: the module does not depend on it) failed at configure or
# link time instead of simply omitting the baseline test.
if(TARGET opencv_perf_triton)
  # `TARGET opencv_cudawarping` is NOT a usable test here: a module's CMakeLists runs in the
  # first pass, before other modules' targets exist, so the target check was false even in a
  # build that had cudawarping enabled. HAVE_opencv_cudawarping is the configure-time answer,
  # and ocv_check_dependencies is the standard way to ask it.
  #
  # Two separate baselines, two separate guards: cv::cuda::resize is in cudawarping and
  # cv::cuda::cvtColor in cudaimgproc, and a build may have either, both or neither. Each
  # missing one omits its own perf test and nothing else.
  ocv_check_dependencies(opencv_cudawarping)
  if(OCV_DEPENDENCIES_FOUND)
    ocv_target_include_modules(opencv_perf_triton opencv_cudawarping)
    ocv_target_link_libraries(opencv_perf_triton LINK_PRIVATE opencv_cudawarping)
    target_compile_definitions(opencv_perf_triton PRIVATE OPENCV_TRITON_PERF_HAVE_CUDAWARPING=1)
  else()
    message(STATUS "triton: opencv_cudawarping not built -- the native CUDA resize baseline "
                   "in perf/ is omitted; the module itself is unaffected")
  endif()
  ocv_check_dependencies(opencv_cudaimgproc)
  if(OCV_DEPENDENCIES_FOUND)
    ocv_target_include_modules(opencv_perf_triton opencv_cudaimgproc)
    ocv_target_link_libraries(opencv_perf_triton LINK_PRIVATE opencv_cudaimgproc)
    target_compile_definitions(opencv_perf_triton PRIVATE OPENCV_TRITON_PERF_HAVE_CUDAIMGPROC=1)
  else()
    message(STATUS "triton: opencv_cudaimgproc not built -- the native CUDA cvtColor baseline "
                   "in perf/ is omitted; the module itself is unaffected")
  endif()
endif()
foreach(_t opencv_test_triton opencv_perf_triton)
  if(TARGET ${_t})
    # the tests compile the portable CPU oracle themselves (test/test_oracle.cpp)
    target_compile_options(${_t} PRIVATE -ffp-contract=off)
    ocv_target_include_directories(${_t} PRIVATE ${_triton_cuda_inc})
    ocv_target_link_libraries(${_t} LINK_PRIVATE ${_triton_cuda_lib})
  endif()
endforeach()
