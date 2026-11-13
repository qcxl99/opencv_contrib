H20 first use: build, install and test {#tutorial_triton_h20_first_use}
===================================

@ingroup triton
@tableofcontents

This walkthrough follows the ordinary OpenCV workflow: choose matching OpenCV and
contrib sources, configure an out-of-source build, install to a private prefix,
then use the public C++ and Python interfaces. It is for Ubuntu x86_64 and an
NVIDIA H20 (`sm_90`), with CUDA Toolkit 12.8 and the validated Triton 3.6.0 ABI.
It is an on-demand guide, not an automatic test or a claim that every command
below has been rerun on the reader's machine.

For the general build procedure see the [OpenCV Linux installation guide](https://github.com/opencv/opencv/blob/5.x/doc/tutorials/introduction/linux_install/linux_install.markdown)
and the [contrib build instructions](https://github.com/opencv/opencv_contrib#how-to-build-opencv-with-extra-modules).
Use their instructions for the selected source revision. This page adds this
module's toolchain, public-entry and cache checks; it does not replace OpenCV's
build system or introduce a different installation package.

Read the module [README](../README.md) for supported inputs and limitations.
CUDA examples use `GpuMat` and caller-supplied streams. They do not intercept
standard `cv::resize` calls, and they do not test the separate HIP device-buffer
interface. A successful first-use run establishes usability of the tested
installation; numerical, performance and contribution acceptance have their own
required coverage.

Prerequisites and source selection
----------------------------------

Have a C++ compiler, CMake, Ninja, Git, Python with development files and venv
support, plus NumPy and the selected Triton package. Typical Ubuntu packages are
`build-essential cmake ninja-build git python3-dev python3-venv libjpeg-dev
libpng-dev`. Install missing packages using the machine's normal administration
procedure; the module is not a reason to replace working system packages.

Choose the exact OpenCV/contrib commits or explicitly identified dirty snapshots
containing this module. A branch name or a source-text match is not an identity
check. Resolve unfinished merge/rebase work before recording a stable acceptance
snapshot. Keep the build, install, virtual environment and writable cache outside
the source trees, including any pinned CPU reference build.

Use one Bash session for the following sections. All commands are manual; opening
a new terminal does not rerun them. Select an idle H20 first. A second visible H20
allows second-device tests; a single-card run must report any corresponding skip.

```bash
set -euo pipefail
read -r -p 'OpenCV source, absolute path: ' H20_MAIN
read -r -p 'opencv_contrib source, absolute path: ' H20_CONTRIB
read -r -p 'Idle H20 index/UUID, comma-separated for a pair: ' H20_DEVICES
export H20_MAIN H20_CONTRIB H20_DEVICES
: "${H20_MAIN:?}" "${H20_CONTRIB:?}" "${H20_DEVICES:?}"
test -f "$H20_MAIN/CMakeLists.txt"
test -f "$H20_CONTRIB/modules/triton/CMakeLists.txt"
H20_ROOT="$(mktemp -d "$HOME/opencv-triton-h20.XXXXXX")"
export H20_ROOT
export H20_BUILD="$H20_ROOT/build"
export H20_INSTALL="$H20_ROOT/install"
export H20_LOG="$H20_ROOT/logs"
export H20_CUDA=/usr/local/cuda-12.8
mkdir -p "$H20_LOG"
uname -a | tee "$H20_LOG/host.txt"
nvidia-smi --query-gpu=index,uuid,name,compute_cap,driver_version,utilization.gpu,memory.used --format=csv \
  | tee "$H20_LOG/devices.txt"
nvidia-smi -i "$H20_DEVICES" | tee "$H20_LOG/selected-devices.txt"
"$H20_CUDA/bin/nvcc" --version | tee "$H20_LOG/nvcc.txt"
export CUDA_VISIBLE_DEVICES="$H20_DEVICES"
for H20_REPO in "$H20_MAIN" "$H20_CONTRIB"; do
  git -C "$H20_REPO" rev-parse HEAD
  git -C "$H20_REPO" status --short
  test -z "$(git -C "$H20_REPO" diff --name-only --diff-filter=U)"
  H20_GIT_DIR="$(git -C "$H20_REPO" rev-parse --absolute-git-dir)"
  test ! -e "$H20_GIT_DIR/MERGE_HEAD"
  test ! -e "$H20_GIT_DIR/CHERRY_PICK_HEAD"
  test ! -d "$H20_GIT_DIR/rebase-merge"
  test ! -d "$H20_GIT_DIR/rebase-apply"
done | tee "$H20_LOG/source-status.txt"
```

Confirm the selected model is H20 and its compute capability is 9.0, not merely
that a CUDA device is visible. `nvidia-smi` reports the driver's supported CUDA
level; the selected `nvcc` reports the Toolkit version. Driver 595.58.03 was used
in the original first-use record; it is a recorded baseline, not a requirement
that all compatible machines use that exact driver. Record changed environments
and validate the affected boundaries.

For an uncommitted contribution, retain staged/unstaged diffs and actual
untracked source files as well as the two SHAs. Do not call a local dirty-tree
run a clean remote-clone reproduction. Never clean or overwrite another task's
source tree to reproduce the examples.

Python environment
------------------

A venv made from Ubuntu's system Python and matching development package is a
simple choice. Another interpreter can work if its headers and development
library are ABI-compatible and actually selected by CMake. A patch-version
number difference alone does not establish an incompatibility.

```bash
command -v git cmake ninja c++ /usr/bin/python3
/usr/bin/python3 -m venv "$H20_ROOT/.venv"
export H20_PY="$H20_ROOT/.venv/bin/python"
unset PYTHONPATH
"$H20_PY" -m pip install --upgrade pip
"$H20_PY" -m pip install numpy triton==3.6.0
"$H20_PY" -m pip freeze > "$H20_LOG/python-packages.txt"
"$H20_PY" - <<'PY'
import pathlib, sys, sysconfig
sys.modules['torch'] = None
import triton
header = pathlib.Path(sysconfig.get_path('include')) / 'Python.h'
library = pathlib.Path(sysconfig.get_config_var('LIBDIR')) / sysconfig.get_config_var('LDLIBRARY')
assert header.is_file(), header
assert library.is_file(), library
assert triton.__version__ == '3.6.0'
print('python:', sys.executable, sys.version)
print('headers:', header, 'library:', library)
print('triton:', triton.__version__, triton.__file__)
PY
H20_PY_INCLUDE="$("$H20_PY" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
export H20_PY_INCLUDE
H20_PY_LIBRARY="$("$H20_PY" -c 'import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_config_var("LIBDIR"))/sysconfig.get_config_var("LDLIBRARY"))')"
export H20_PY_LIBRARY
```

Do not install `opencv-python` or PyTorch for this walkthrough. The former can
hide the extension being built; the latter is not a module dependency. Existing
packages need not be uninstalled blindly: use the import checks below. The
Torch-blocked import is only an import check, not proof of cold compilation.
Triton versions other than the validated launch ABI are not silently accepted.

Configure, build and install
---------------------------

This is a headless first-use configuration. The disabled UI/media options reduce
incidental dependencies; applications may enable them normally. The test-module
selectors limit this build to Triton tests, while the normal build target builds
all configured libraries before installation. The SSE3/no-IPP
settings keep this x86_64 test build comparable to the characterised CPU
configuration. Resize's C++ tests still use their portable oracle; these settings
do not promise all CPU functions are build-independent.

```bash
cmake -S "$H20_MAIN" -B "$H20_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$H20_INSTALL" \
  -DOPENCV_EXTRA_MODULES_PATH="$H20_CONTRIB/modules" \
  -DBUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,cudev,triton,python3,python_bindings_generator,ts \
  -DBUILD_opencv_triton=ON -DTRITON_BACKEND=cuda -DWITH_CUDA=ON \
  -DCUDA_TOOLKIT_ROOT_DIR="$H20_CUDA" -DCUDA_ARCH_BIN=9.0 -DCUDA_ARCH_PTX= \
  -DWITH_IPP=OFF -DCPU_BASELINE=SSE3 -DCPU_DISPATCH= -DWITH_OPENCL=OFF \
  -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_QT=OFF \
  -DWITH_OPENEXR=OFF -DWITH_NVCUVID=OFF -DWITH_NVCUVENC=OFF \
  -DPYTHON3_EXECUTABLE="$H20_PY" -DPYTHON3_INCLUDE_DIR="$H20_PY_INCLUDE" \
  -DPYTHON3_LIBRARY="$H20_PY_LIBRARY" \
  -DTRITON_PYTHON_EXECUTABLE="$H20_PY" -DTRITON_PYTHONPATH= \
  -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON -DINSTALL_TESTS=ON \
  -DOPENCV_BUILD_TEST_MODULES_LIST=triton -DOPENCV_BUILD_PERF_TEST_MODULES_LIST=triton \
  -DBUILD_opencv_apps=OFF \
  2>&1 | tee "$H20_LOG/configure.log"
cmake --build "$H20_BUILD" --parallel "${H20_JOBS:-8}" \
  2>&1 | tee "$H20_LOG/build.log"
cmake --install "$H20_BUILD" 2>&1 | tee "$H20_LOG/install.log"
cp "$H20_BUILD/CMakeCache.txt" "$H20_LOG/CMakeCache.txt"
```

Check the configuration selected CUDA12.8, sm_90, the CUDA backend, Triton3.6.0,
triton/python3 and the intended interpreter/include/library paths. Confirm the
three example binaries and test/perf programs exist. `Configuring done` alone
does not prove Python bindings were built; `python_bindings_generator` must be
available. Use suitable build concurrency for the host's memory.

`cudawarping` and `cudaimgproc` are not production dependencies. If native CUDA
performance comparisons are part of the task, include them in BUILD_LIST and
record the actual optional baselines/NPP build. Missing a baseline is not a
speedup result. Nothing in this procedure requests AOT binaries.

Use the installed Python extension
---------------------------------

The build loader, build-tree extension and installed extension have different
legitimate layouts. Here, select the unique extension inside the private install
prefix. A path containing `site-packages` is not wrong if it is inside that
prefix. Do not accept an unrelated loader's `__init__.py` as sufficient proof.

```bash
unset OPENCV_TRITON_RESOURCES OPENCV_TRITON_COMPILER
export OPENCV_TRITON_PYTHON="$H20_PY"
export OPENCV_TRITON_PYTHONPATH=
export OPENCV_TRITON_CACHE_DIR="$H20_ROOT/cache-main"
export LD_LIBRARY_PATH="$H20_INSTALL/lib:$H20_CUDA/lib64"
export PYTHONNOUSERSITE=1
H20_CV2_DIR="$("$H20_PY" - <<'PY'
import os, pathlib
files = list(pathlib.Path(os.environ['H20_INSTALL']).rglob('cv2*.so'))
assert len(files) == 1, files
print(files[0].parent)
PY
)"
export H20_CV2_DIR
export PYTHONPATH="$H20_CV2_DIR:$H20_MAIN/modules/python/test"
"$H20_PY" - <<'PY' | tee "$H20_LOG/import-identity.log"
import os, pathlib, cv2 as cv
prefix = pathlib.Path(os.environ['H20_INSTALL']).resolve()
assert pathlib.Path(cv.__file__).resolve().is_relative_to(prefix), cv.__file__
assert cv.cuda.getCudaEnabledDeviceCount() > 0
cv.cuda.setDevice(0)
assert cv.triton.haveBackend() and cv.triton.backendName() == 'cuda-jit'
for name in ('resize', 'cvtColor', 'warpAffine'):
    assert hasattr(cv.triton, name), name
    print(name, getattr(cv.triton, name).__doc__)
print('cv2:', cv.__file__, 'OpenCV:', cv.__version__)
print(cv.getBuildInformation())
libs = {line.split(maxsplit=5)[5] for line in pathlib.Path('/proc/self/maps').read_text().splitlines()
        if 'libopencv_' in line and len(line.split(maxsplit=5)) == 6}
assert any('libopencv_triton.' in p for p in libs), libs
assert all(pathlib.Path(p).resolve().is_relative_to(prefix) for p in libs), libs
print('loaded OpenCV libraries:', sorted(libs))
PY
```

These overrides apply to this test shell, not global configuration. Existing
resource/compiler overrides should be recorded before clearing them. Keep a
matching interpreter for a cold cache. Hot-path C++ execution does not eliminate
the cold-cache Python/Triton dependency.

Public C++ samples and Python pipeline
-------------------------------------

```bash
"$H20_BUILD/bin/example_triton_resize" 2>&1 | tee "$H20_LOG/sample-resize.log"
"$H20_BUILD/bin/example_triton_cvtcolor" 2>&1 | tee "$H20_LOG/sample-cvtcolor.log"
"$H20_BUILD/bin/example_triton_warpaffine" 2>&1 | tee "$H20_LOG/sample-warpaffine.log"
cat > "$H20_ROOT/demo.py" <<'PY'
import cv2 as cv
import numpy as np
stream = cv.cuda_Stream()
host = np.random.default_rng(7).integers(0, 256, (96, 128, 3), dtype=np.uint8)
src = cv.cuda_GpuMat()
src.upload(host, stream)
small = cv.triton.resize(src, (64, 48), interpolation=cv.INTER_LINEAR, stream=stream)
gray = cv.triton.cvtColor(small, cv.COLOR_BGR2GRAY, stream=stream)
matrix = np.eye(2, 3, dtype=np.float64)
warped = cv.triton.warpAffine(gray, matrix, (64, 48), flags=cv.INTER_NEAREST, stream=stream)
stream.waitForCompletion()
result = warped.download()
assert result.shape == (48, 64) and result.dtype == np.uint8
print(cv.triton.backendName(), host.shape, '->', result.shape, result.dtype)
PY
"$H20_PY" "$H20_ROOT/demo.py" 2>&1 | tee "$H20_LOG/python-demo.log"
```

The required Python cvtColor arguments are `(src, code)`, with optional `dst` and
`stream` according to the generated signature printed above. C++ uses
`cv::triton::cvtColor(src, dst, code, stream)`. Retain inputs/outputs until the
stream completes. The pipeline shows connectivity and ordering, not complete
numerical agreement with an arbitrary CPU build. Correctness is tested below.

OpenCV tests and a small perf check
----------------------------------

`INSTALL_TESTS=ON` installs the native test programs. Resolve their actual paths
instead of assuming all revisions install into exactly the same subdirectory.

```bash
H20_TEST_BIN="$("$H20_PY" - <<'PY'
import os, pathlib
files = list(pathlib.Path(os.environ['H20_INSTALL']).rglob('opencv_test_triton'))
assert len(files) == 1, files
print(files[0])
PY
)"
export H20_TEST_BIN
H20_PERF_BIN="$("$H20_PY" - <<'PY'
import os, pathlib
files = list(pathlib.Path(os.environ['H20_INSTALL']).rglob('opencv_perf_triton'))
assert len(files) == 1, files
print(files[0])
PY
)"
export H20_PERF_BIN
ldd "$H20_TEST_BIN" | tee "$H20_LOG/test-linkage.log"
"$H20_TEST_BIN" --gtest_list_tests > "$H20_LOG/accuracy-inventory.txt"
"$H20_TEST_BIN" --gtest_output="xml:$H20_LOG/accuracy.xml" \
  2>&1 | tee "$H20_LOG/accuracy.log"
"$H20_PY" "$H20_CONTRIB/modules/triton/misc/python/test/test_triton.py" -v \
  2>&1 | tee "$H20_LOG/python-tests.log"
"$H20_PERF_BIN" --gtest_list_tests > "$H20_LOG/perf-inventory.txt"
# Inspect the inventory first. Adjust if the selected revision renamed these tests.
"$H20_PERF_BIN" \
  '--gtest_filter=*TritonResize/0:*TritonCvtColor/0:*TritonWarpAffine/0' \
  --perf_min_samples=3 --perf_max_samples=5 --perf_force_samples=3 --perf_write_sanity \
  --gtest_output="xml:$H20_LOG/perf.xml" \
  2>&1 | tee "$H20_LOG/perf-smoke.log"
"$H20_PY" - <<'PY' | tee "$H20_LOG/test-summary.log"
import os, pathlib, xml.etree.ElementTree as ET
root = pathlib.Path(os.environ['H20_LOG'])
for name in ('accuracy', 'perf'):
    tree = ET.parse(root / (name + '.xml')).getroot()
    tests = int(tree.attrib.get('tests', 0))
    failures = int(tree.attrib.get('failures', 0)) + int(tree.attrib.get('errors', 0))
    skipped = sum(1 for c in tree.iter('testcase') if c.find('skipped') is not None
                  or c.attrib.get('status') == 'notrun')
    assert tests > 0 and failures == 0, (name, tree.attrib)
    if name == 'perf':
        assert tests == 3 and skipped == 0, (name, tree.attrib, skipped)
    print(name, 'reported=', tests, 'failures=', failures, 'skipped/notrun=', skipped)
PY
```

Check accuracy's actual test list against the recorded inventory and advertised
scope. Report skipped/disabled cases separately, including single-card
second-device skips or HIP-only Python tests on CUDA. Zero tests, help output,
missing/stale XML or all relevant tests skipped are not a pass. Do not use an old
record's test count as the new revision's required denominator.

The three perf cases and sampling values above are a **smoke budget**, not a
complete workload or a performance threshold. They include the public call and
stream completion; they are not graph replay or pure-kernel timing. Full perf
and other measurements should be explicitly selected when required. A missing
opencv_extra warning is harmless only when the selected tests do not require
that data and the actual tests pass; required-data skips must be investigated.

External C++ consumer and cache behaviour
----------------------------------------

An ordinary C++ application finds OpenCV from the install prefix, not private
module headers or build directories. The constant-input check below is a public
API/deployment check, not the portable numerical oracle.

```bash
mkdir -p "$H20_ROOT/consumer"
cat > "$H20_ROOT/consumer/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.16)
project(h20_first_use LANGUAGES CXX)
find_package(OpenCV REQUIRED COMPONENTS core imgproc triton)
add_executable(consumer main.cpp)
target_include_directories(consumer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(consumer PRIVATE ${OpenCV_LIBS})
CMAKE
cat > "$H20_ROOT/consumer/main.cpp" <<'CPP'
#include <opencv2/triton.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
int main() try
{
    if (!cv::triton::haveBackend()) return 77;
    cv::Mat host(48, 64, CV_8UC3, cv::Scalar(11, 23, 47));
    cv::cuda::GpuMat src(host), small, gray, warped;
    cv::cuda::Stream stream;
    const cv::Mat matrix = cv::Mat::eye(2, 3, CV_64F);
    cv::triton::resize(src, small, cv::Size(32, 24), 0, 0, cv::INTER_LINEAR, stream);
    cv::triton::cvtColor(small, gray, cv::COLOR_BGR2GRAY, stream);
    cv::triton::warpAffine(gray, warped, matrix, cv::Size(32, 24), cv::INTER_NEAREST,
                           cv::BORDER_REPLICATE, cv::Scalar(), stream);
    stream.waitForCompletion();
    cv::Mat got, reference;
    warped.download(got);
    cv::cvtColor(host, reference, cv::COLOR_BGR2GRAY);
    const cv::Mat expected(got.size(), got.type(), cv::Scalar::all(reference.at<uchar>(0, 0)));
    const double error = cv::norm(got, expected, cv::NORM_INF);
    std::cout << cv::triton::backendName() << " pipeline maxdiff=" << error << '\n';
    return error == 0 ? 0 : 1;
}
catch (const cv::Exception& e) { std::cerr << e.what() << '\n'; return 2; }
CPP
H20_CONFIG="$("$H20_PY" - <<'PY'
import os, pathlib
files = list(pathlib.Path(os.environ['H20_INSTALL']).rglob('OpenCVConfig.cmake'))
assert len(files) == 1, files
print(files[0].parent)
PY
)"
export H20_CONFIG
cmake -S "$H20_ROOT/consumer" -B "$H20_ROOT/consumer-build" \
  -DOpenCV_DIR="$H20_CONFIG" 2>&1 | tee "$H20_LOG/consumer-configure.log"
cmake --build "$H20_ROOT/consumer-build" 2>&1 | tee "$H20_LOG/consumer-build.log"
export H20_CONSUMER="$H20_ROOT/consumer-build/consumer"
export OPENCV_TRITON_CACHE_DIR="$H20_ROOT/cache-deployment"
test ! -e "$OPENCV_TRITON_CACHE_DIR"
"$H20_CONSUMER" 2>&1 | tee "$H20_LOG/consumer-cold.log"
OPENCV_TRITON_PYTHON=/missing/python "$H20_CONSUMER" \
  2>&1 | tee "$H20_LOG/consumer-hot-no-python.log"
"$H20_PY" - <<'PY' | tee "$H20_LOG/consumer-cold-no-python.log"
import os, pathlib, subprocess
cache = pathlib.Path(os.environ['H20_ROOT']) / 'cache-toolchain-unavailable'
assert not cache.exists(), cache
env = dict(os.environ, OPENCV_TRITON_CACHE_DIR=str(cache), OPENCV_TRITON_PYTHON='/missing/python')
p = subprocess.run([os.environ['H20_CONSUMER']], env=env, capture_output=True, text=True)
print(p.stdout, p.stderr, 'returncode=', p.returncode)
assert p.returncode == 2 and 'toolchain_unavailable' in (p.stdout + p.stderr)
PY
```

These are distinct processes and a new OpenCV artifact cache: first use must
produce verified artifacts, disk reuse must work without an interpreter, and an
empty cache without the toolchain must fail clearly. This does not isolate every
underlying compiler cache and does not establish a fully compiler-cold latency.
No new startup-time threshold is imposed here.

Completion and later sessions
-----------------------------

Record the two source revisions and actual dirty bytes, selected GPU/driver and
Toolkit, Python/Triton paths and versions, CMakeCache, commands/filters, loaded
extension/libraries, raw logs/XML, source/resource/artifact hashes and cache state.
Do not use modification times to establish identity: Python kernels and helpers
are runtime inputs too. If sources change, assess the affected checks and bind
new evidence rather than relabelling the old result.

A first-use result should separately state build/import, examples and public
calls, accuracy/Python results, perf smoke, and cold/hot/error behaviour. List
skips and untested claims. Contributions/deliveries may additionally require:

- install/resource/header checks and a relocated-prefix consumer;
- the module's ROI, output reuse, context, stream, graph and in-flight lifetime
  coverage for every claimed public entry/backend;
- the complete agreed performance workload and applicable host/device/graph,
  startup and working-set measurements if claiming performance;
- reference-specific numerical evidence, not a promise about every CPU build;
- the target project's contribution checklist and final code review.

A first-use pass is not full qualification, a speedup result, HIP acceptance or
maintainer approval. Read the current source's tests/contracts and contribution
requirements for those decisions. In particular, synchronise C++ and Python
positive/refusal tests when functionality expands; do not retain stale rejection
assertions or simply remove them to get a pass.

For a new terminal, restore the chosen installation and interpreter and run the
application only. Do not put this guide in a login script or rerun the tests:

```bash
read -r -p 'Saved first-use directory: ' H20_ROOT
read -r -p 'Idle H20 index/UUID for this application session: ' H20_DEVICES
export H20_ROOT H20_DEVICES
export H20_INSTALL="$H20_ROOT/install"
export H20_PY="$H20_ROOT/.venv/bin/python"
export H20_CUDA=/usr/local/cuda-12.8
export CUDA_VISIBLE_DEVICES="$H20_DEVICES"
export LD_LIBRARY_PATH="$H20_INSTALL/lib:$H20_CUDA/lib64"
export PYTHONNOUSERSITE=1
H20_CV2_DIR="$("$H20_PY" - <<'PY'
import os, pathlib
files = list(pathlib.Path(os.environ['H20_INSTALL']).rglob('cv2*.so'))
assert len(files) == 1, files
print(files[0].parent)
PY
)"
export H20_CV2_DIR
export PYTHONPATH="$H20_CV2_DIR"
unset OPENCV_TRITON_RESOURCES OPENCV_TRITON_COMPILER
export OPENCV_TRITON_PYTHON="$H20_PY"
export OPENCV_TRITON_PYTHONPATH=
export OPENCV_TRITON_CACHE_DIR="$H20_ROOT/cache-deployment"
"$H20_PY" "$H20_ROOT/demo.py"
```
