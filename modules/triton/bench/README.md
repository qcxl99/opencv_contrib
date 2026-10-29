# bench/ -- CUDA-only, from kernelforge-lab

**Platform: CUDA (NVIDIA, e.g. H20) only.** Nothing in this directory runs
on the HIP/BW1000 backend (`src/backend/hip/resize.cpp`,
`TRITON_BACKEND_HIP`). These scripts call kernelforge-lab's own CUDA Triton
kernel (`operators/resize/kernel.py`) and `cv::cuda::resize`
(`operators/resize/oracle_cuda.py`) through kernelforge-lab's own
ctypes/AOT layer and `cases.yaml` suite -- not through this module's
`cv::triton` C++ API, and not through this module's CMake build at all.

They are carried in here as reference/companion material from the sibling
`kernelforge-lab` project (correctness + performance scripts for the same
resize kernel this module's CUDA backend wraps), not as part of this
module's own test/perf suite. For this module's own CUDA test/perf
coverage (CMake-driven, `ocv_add_accuracy_tests`/`ocv_add_perf_tests`), see
`../test/test_resize.cpp` and `../perf/perf_resize.cpp` -- those are also
CUDA-only today; the HIP backend currently has no test or perf coverage at
all (tracked separately, not addressed by this directory).

## Running

Requires a `kernelforge-lab` checkout. If this directory is not itself
inside one (it is not, once copied here), point `KERNELFORGE_LAB_ROOT` at
one:

```bash
export KERNELFORGE_LAB_ROOT=/path/to/kernelforge-lab
export LD_LIBRARY_PATH=/opt/ocv-cuda/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib
python3 bench_correctness_linear.py
python3 bench_performance_area.py
python3 bench_all.py          # everything, correctness then performance
```

Each script's own module docstring documents which `cases.yaml` suite and
tolerance criterion it uses.
