HIP Python device images {#tutorial_triton_hip_python}
========================

The HIP backend provides `cv2.triton.HipImage` and `cv2.triton.HipStream`.
`resize`, `cvtColor` and `warpAffine` accept device images and return device
images, so a pipeline keeps its intermediate data on the GPU. NumPy upload and
download are blocking convenience transfers. PyTorch is not required.

Build OpenCV with the HIP backend and Python bindings as described in
@ref tutorial_triton_bw1000_first_use. Use the matching installed `cv2` package
and OpenCV libraries. `cv.triton.backendName()` identifies the available
backend; `cv.triton.haveBackend()` checks its availability.

Device pipeline
---------------

```python
import cv2 as cv
import numpy as np

image = np.zeros((480, 640, 3), dtype=np.uint8)
src = cv.triton.HipImage(device=0)
src.upload(image)
stream = cv.triton.HipStream(device=0)

small = cv.triton.resize(src, (320, 240),
                         interpolation=cv.INTER_AREA, stream=stream)
gray = cv.triton.cvtColor(small, cv.COLOR_BGR2GRAY, stream=stream)
M = np.array([[1, 0, 8], [0, 1, 4]], dtype=np.float64)
warped = cv.triton.warpAffine(gray, M, (320, 240),
                             flags=cv.INTER_NEAREST, stream=stream)
result = warped.download()
```

`download()` waits for pending image work and returns a NumPy array. No
additional `waitForCompletion()` is required before downloading. The complete
sample is `samples/hip/device.py`.

Images and transfers
--------------------

`HipImage(device=0)` constructs an empty image on a visible HIP device. Object
creation and operations preserve the caller's current HIP context. The device
is fixed for the object's lifetime.

`upload(array)` allocates or reuses pitched device storage and copies a nonempty
2-D image. It accepts NumPy `uint8`, `uint16` and `float32` arrays with one to
four channels. Multichannel arrays use `(height, width, channels)` order.
Strided host images are accepted. Operator depth and channel restrictions still
apply; see @ref tutorial_triton_resize, @ref tutorial_triton_cvtcolor and
@ref tutorial_triton_warpaffine.

`upload()` finishes the host-to-device copy before returning, so the host array
can then be modified or released. `download()` returns an independent host
array of matching size and dtype. Single-channel output has shape
`(height, width)`; multichannel output has shape `(height, width, channels)`.

Use `empty()`, `size()`, `type()`, `depth()`, `channels()`, `device()` and `step()`
to inspect an image. `size()` returns `(width, height)`, `type()` and `depth()`
use OpenCV type constants, and `step()` is the row pitch in bytes.

Output reuse and regions
------------------------

Each operator accepts an optional `dst` and returns the output device image:

```python
dst = cv.triton.HipImage(device=src.device())
result = cv.triton.resize(src, (320, 240), dst=dst, stream=stream)
```

Matching size and type reuse the destination storage, including ROI storage.
Otherwise ordinary calls allocate the required output. For
`warpAffine(..., borderMode=cv.BORDER_TRANSPARENT)`, `dst` must already contain
an image of the required size and type, initialized by `upload()` or an earlier
operation. Transparent borders preserve or use those existing pixels as
specified in @ref tutorial_triton_warpaffine; an empty or mismatched output is
rejected.

`image.roi((x, y, width, height))` returns a view with shared device storage:

```python
view = src.roi((16, 8, 128, 96))
small = cv.triton.resize(view, (64, 48), stream=stream)
```

The ROI must be nonempty and contained in the image. It retains the allocation
and remains valid after its parent object is released. Writing a view changes
the corresponding parent region. Source and destination pixel storage must not
overlap; disjoint ROIs may share an allocation. In-place operations are
unsupported.

Streams and lifetime
--------------------

Operators enqueue work on the supplied `HipStream`. Omitting `stream` or
passing `None` selects the source device's default stream. Source, destination
and explicit stream must belong to the same device; mismatches are rejected.
Device indices refer to the devices exposed by the platform's visibility
settings.

Shared storage tracks outstanding reads and writes. Operators insert image
dependencies across streams automatically, including for ROI aliases:

```python
producer = cv.triton.HipStream(device=0)
consumer = cv.triton.HipStream(device=0)
small = cv.triton.resize(src, (320, 240), stream=producer)
gray = cv.triton.cvtColor(small, cv.COLOR_BGR2GRAY, stream=consumer)
result = gray.download()
```

`consumer.waitFor(producer)` orders subsequent consumer work after work already
queued in the producer, without blocking the host. It is available for explicit
ordering; ordinary image dependencies do not require it.
`stream.waitForCompletion()` blocks until queued work completes, and
`stream.queryIfComplete()` checks completion without blocking.

Device objects keep storage alive for queued operations. Releasing the last
owner of an allocation waits for its outstanding work before freeing it;
releasing a stream waits for its queued work. Such destruction can block the
host. Keep reusable images and streams alive to avoid synchronization at each
pipeline stage.

The owning API does not import external tensor pointers or support HIP graph
capture. C++ applications managing their own allocations and native streams can
use `opencv2/triton/device.hpp`. The owning classes are also available in C++
from `opencv2/triton/hip.hpp`. GpuMat overloads require the CUDA backend;
HipImage and HipStream require the HIP backend.
