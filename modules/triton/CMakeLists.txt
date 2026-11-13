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
# Backends (TRITON_BACKEND = auto | cuda | hip), Linux only:
#   cuda  CUDA driver API (src/runtime/cuda_runtime.cpp), kernels/*_cpu.py, the cuda::GpuMat
#         entries of opencv2/triton.hpp, and the GpuMat-based tests / perf / samples.
#   hip   HIP on Hygon DTK / ROCm (src/runtime/hip_runtime.cpp, BW1000 = gfx936). Kernels are
#         kernels/*_cpu_hip.py where the CUDA source needs a derivation (tools/
#         make_kernels_cpu_hip.py; a stale copy fails the configure), else the CUDA source as
#         written. The entries are opencv2/triton/device.hpp (device buffers + a hip stream);
#         the GpuMat entries exist and raise, since OpenCV has no cv::cuda here. Accuracy is
#         test/hip/. Needs a Triton that targets hip (flagtree, not upstream) and ROCM_PATH.
#   auto  cuda where the OpenCV build has CUDA (unchanged), else hip if a HIP toolkit is found,
#         else the module is disabled. A tree whose CUDA is a compatibility layer (WITH_CUDA=ON
#         on a BW1000 box) must say -DTRITON_BACKEND=hip.
# Windows and macOS are disabled here, not silently half-built.
set(BUILD_opencv_triton_INIT OFF)
set(the_description "Triton-backed operations (runtime JIT)")
# Register first: OpenCV returns during discovery and skips disabled modules,
# so ordinary builds never probe Python/Triton or require a HIP/CUDA toolkit.
ocv_add_module(triton opencv_core opencv_imgproc WRAP python)

set(TRITON_BACKEND "auto" CACHE STRING "Device backend of the triton module: auto, cuda or hip")
set_property(CACHE TRITON_BACKEND PROPERTY STRINGS auto cuda hip)

# HIP toolkit. TRITON_ROCM_PATH is the toolkit root and doubles as the default ROCM_PATH the
# compile helper's compiler needs at run time; the environment's ROCM_PATH still wins then.
set(TRITON_ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm/DTK root, the compile helper's default ROCM_PATH (hip backend)")
find_path(TRITON_HIP_INCLUDE_DIR hip/hip_runtime_api.h
          HINTS "${TRITON_ROCM_PATH}" ENV ROCM_PATH ENV HIP_PATH
          PATHS /opt/dtk /opt/rocm
          PATH_SUFFIXES include hip/include)
find_library(TRITON_HIP_LIBRARY amdhip64
             HINTS "${TRITON_ROCM_PATH}" ENV ROCM_PATH ENV HIP_PATH
             PATHS /opt/dtk /opt/rocm
             PATH_SUFFIXES lib lib64 hip/lib)
mark_as_advanced(TRITON_HIP_INCLUDE_DIR TRITON_HIP_LIBRARY)
set(_triton_have_hip OFF)
if(TRITON_HIP_INCLUDE_DIR AND TRITON_HIP_LIBRARY)
  set(_triton_have_hip ON)
endif()

set(_triton_backend "none")
if(TRITON_BACKEND STREQUAL "auto")
  if(HAVE_CUDA)
    set(_triton_backend "cuda")
  elseif(_triton_have_hip)
    set(_triton_backend "hip")
  endif()
elseif(TRITON_BACKEND STREQUAL "cuda")
  if(HAVE_CUDA)
    set(_triton_backend "cuda")
  else()
    message(WARNING "triton: TRITON_BACKEND=cuda but this OpenCV build has no CUDA; the module is disabled")
  endif()
elseif(TRITON_BACKEND STREQUAL "hip")
  if(_triton_have_hip)
    set(_triton_backend "hip")
  else()
    message(WARNING "triton: TRITON_BACKEND=hip but hip/hip_runtime_api.h or libamdhip64 was not found "
                    "(set TRITON_ROCM_PATH or ROCM_PATH); the module is disabled")
  endif()
else()
  message(FATAL_ERROR "triton: TRITON_BACKEND must be auto, cuda or hip (got '${TRITON_BACKEND}')")
endif()

if(IOS OR WINRT OR WIN32 OR APPLE OR _triton_backend STREQUAL "none")
  ocv_module_disable(triton)
endif()
message(STATUS "triton: backend ${_triton_backend}")

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
    "${CMAKE_CURRENT_LIST_DIR}/src/device_entries.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/python_entries.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/reference_cpu.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/provider.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/cvtcolor.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/warpaffine.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/compile_protocol.cpp")
if(_triton_backend STREQUAL "hip")
  # The GpuMat entries need cv::cuda, which a hip build of OpenCV does not have: this file
  # defines the same symbols and raises, pointing at opencv2/triton/device.hpp.
  list(APPEND _triton_srcs
       "${CMAKE_CURRENT_LIST_DIR}/src/gpumat_adapter_unavailable.cpp"
       "${CMAKE_CURRENT_LIST_DIR}/src/runtime/hip_runtime.cpp")
  set(_triton_kernel_files resize_cpu_hip.py cvtcolor_cpu_hip.py warpaffine_cpu_hip.py)
else()
  list(APPEND _triton_srcs
       "${CMAKE_CURRENT_LIST_DIR}/src/cuda_resize_adapter.cpp"
       "${CMAKE_CURRENT_LIST_DIR}/src/runtime/cuda_runtime.cpp")
  set(_triton_kernel_files resize_cpu.py cvtcolor_cpu.py warpaffine_cpu.py)
endif()
# Installed headers: only the public umbrella. src/core.hpp is deliberately NOT here -- it is a
# raw device-pointer layer the plan code shares with the GpuMat adapter, not an API for callers.
file(GLOB _triton_hdrs "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/triton.hpp"
                       "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/triton/device.hpp")
file(GLOB _triton_int_hdrs "${CMAKE_CURRENT_LIST_DIR}/src/*.hpp" "${CMAKE_CURRENT_LIST_DIR}/src/runtime/*.hpp")

# imgproc: the interpolation enums. cudawarping is not a dependency of the implementation; it is
# the performance baseline of perf/ only (a different computation: cv::cuda semantics).
ocv_set_module_sources(HEADERS ${_triton_hdrs} SOURCES ${_triton_srcs} ${_triton_int_hdrs})
ocv_module_include_directories()
ocv_create_module()

if(_triton_backend STREQUAL "cuda")
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
else()
  # --- HIP (DTK / ROCm) -----------------------------------------------------------------
  # SYSTEM: the toolkit headers come after OpenCV's own, and their deprecation noise stays out.
  # __HIP_PLATFORM_AMD__ is what hipcc defines implicitly and a plain C++ compiler must be told.
  set(_triton_rocm "${TRITON_ROCM_PATH}")
  if(NOT _triton_rocm)
    get_filename_component(_triton_rocm "${TRITON_HIP_INCLUDE_DIR}" DIRECTORY)
  endif()
  target_include_directories(${the_module} SYSTEM PRIVATE "${TRITON_HIP_INCLUDE_DIR}")
  ocv_target_link_libraries(${the_module} LINK_PRIVATE "${TRITON_HIP_LIBRARY}" ${CMAKE_DL_LIBS})
  ocv_target_compile_definitions(${the_module} PRIVATE
    __HIP_PLATFORM_AMD__=1
    OPENCV_TRITON_ROCM_PATH_DEFAULT="${_triton_rocm}")
  message(STATUS "triton: hip ${TRITON_HIP_INCLUDE_DIR}, ${TRITON_HIP_LIBRARY}, ROCM_PATH default ${_triton_rocm}")

  # The hip kernels are derived from the CUDA sources; a stale copy would serve kernels that
  # differ from what the CUDA source defines, so drift fails the configure.
  if(TRITON_PYTHON_EXECUTABLE)
    execute_process(COMMAND "${TRITON_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tools/make_kernels_cpu_hip.py" --check
                    RESULT_VARIABLE _triton_hip_rc OUTPUT_QUIET ERROR_VARIABLE _triton_hip_err)
    if(NOT _triton_hip_rc EQUAL 0)
      message(FATAL_ERROR "triton: ${_triton_hip_err}")
    endif()
    # Same rule for the runtime: src/runtime/hip_runtime.cpp is generated from cuda_runtime.cpp.
    execute_process(COMMAND "${TRITON_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tools/port_hip_runtime.py" --check
                    RESULT_VARIABLE _triton_hip_rc OUTPUT_QUIET ERROR_VARIABLE _triton_hip_err)
    if(NOT _triton_hip_rc EQUAL 0)
      message(FATAL_ERROR "triton: ${_triton_hip_err}")
    endif()
  else()
    message(WARNING "triton: no Python to check that kernels/*_cpu_hip.py and src/runtime/hip_runtime.cpp "
                    "are current (run tools/make_kernels_cpu_hip.py --check, tools/port_hip_runtime.py --check)")
  endif()
endif()

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
set(_triton_kernel_paths "")
foreach(_k ${_triton_kernel_files})
  list(APPEND _triton_kernel_paths "${CMAKE_CURRENT_LIST_DIR}/kernels/${_k}")
endforeach()
install(FILES ${_triton_kernel_paths} DESTINATION "${_triton_res_install}/kernels" COMPONENT libs)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/tools/compile_helper.py" DESTINATION "${_triton_res_install}/tools" COMPONENT libs)

# --- tests / perf / samples -------------------------------------------------------------
if(_triton_backend STREQUAL "cuda")
  # OpenCV recursively globs test/ by default; HIP's standalone main/toolkit
  # sources must not be compiled into the CUDA gtest executable.
  file(GLOB _triton_cuda_test_sources "${CMAKE_CURRENT_LIST_DIR}/test/*.cpp"
                                     "${CMAKE_CURRENT_LIST_DIR}/test/*.hpp")
  ocv_add_accuracy_tests(FILES "Src" ${_triton_cuda_test_sources})
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
else()
  # test/, perf/ and samples/ are written against cuda::GpuMat / cuda::Stream and the CUDA driver
  # API, so they are not built here. The hip backend's accuracy test drives the device entries
  # (opencv2/triton/device.hpp) on real hip buffers, against the same portable oracle and the
  # same contract bounds (test/test_helpers.hpp owns them; the test reads them from there).
  if(BUILD_EXAMPLES)
    add_executable(example_triton_device "${CMAKE_CURRENT_LIST_DIR}/samples/device.cpp")
    ocv_target_include_modules(example_triton_device opencv_core opencv_imgproc ${the_module})
    target_include_directories(example_triton_device SYSTEM PRIVATE "${TRITON_HIP_INCLUDE_DIR}")
    target_compile_definitions(example_triton_device PRIVATE __HIP_PLATFORM_AMD__=1)
    target_link_libraries(example_triton_device PRIVATE opencv_core opencv_imgproc ${the_module} "${TRITON_HIP_LIBRARY}")
    set_target_properties(example_triton_device PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")
  endif()
  option(TRITON_BUILD_HIP_TEST "Build opencv_test_triton_hip, the hip backend's accuracy test" ON)
  if(TRITON_BUILD_HIP_TEST)
    set(_t opencv_test_triton_hip)
    add_executable(${_t} "${CMAKE_CURRENT_LIST_DIR}/test/hip/test_hip.cpp" "${CMAKE_CURRENT_LIST_DIR}/test/test_oracle.cpp")
    ocv_target_include_modules(${_t} opencv_core opencv_imgproc ${the_module})
    target_include_directories(${_t} SYSTEM PRIVATE "${TRITON_HIP_INCLUDE_DIR}")
    target_compile_definitions(${_t} PRIVATE __HIP_PLATFORM_AMD__=1)
    # as the other tests: the oracle must not be contracted into FMAs
    target_compile_options(${_t} PRIVATE -ffp-contract=off)
    target_link_libraries(${_t} PRIVATE opencv_core opencv_imgproc ${the_module} "${TRITON_HIP_LIBRARY}" ${CMAKE_DL_LIBS})
    set_target_properties(${_t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}" FOLDER "tests accuracy")
  endif()
endif()
