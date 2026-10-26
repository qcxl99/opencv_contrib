# Build-time Triton compilation. Running the kernel definition is a BUILD
# action, not a static check, and it must not happen on the call path.
function(compile_triton_resize)
  cmake_parse_arguments(T "" "KERNEL;TOOL;OUTPUT_DIR;MANIFEST" "" ${ARGN})

  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(NOT Python3_FOUND)
    message(FATAL_ERROR "triton: WITH_TRITON=ON requires a Python 3 interpreter at build time")
  endif()

  execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import triton; print(triton.__version__)"
                  OUTPUT_VARIABLE TRITON_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE
                  RESULT_VARIABLE TRITON_PROBE ERROR_QUIET)
  if(NOT TRITON_PROBE EQUAL 0)
    message(FATAL_ERROR "triton: Triton is not importable by ${Python3_EXECUTABLE}")
  endif()
  message(STATUS "triton: compiling kernels with Triton ${TRITON_VERSION}")

  file(MAKE_DIRECTORY "${T_OUTPUT_DIR}")
  add_custom_command(
    OUTPUT "${T_MANIFEST}"
    COMMAND "${Python3_EXECUTABLE}" "${T_TOOL}"
            --kernel "${T_KERNEL}"
            --out-dir "${T_OUTPUT_DIR}"
            --manifest "${T_MANIFEST}"
    DEPENDS "${T_KERNEL}" "${T_TOOL}"
    COMMENT "triton: generating resize artifacts"
    VERBATIM)
  add_custom_target(opencv_triton_kernels DEPENDS "${T_MANIFEST}")
endfunction()
