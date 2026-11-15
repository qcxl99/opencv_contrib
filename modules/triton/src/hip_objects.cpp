// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
#include "opencv2/triton/hip.hpp"
#include "opencv2/triton/device.hpp"
#include "entry.hpp"
#include <mutex>
#include <map>

#ifdef OPENCV_TRITON_HAVE_HIP_ADAPTER
#include <hip/hip_runtime_api.h>
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace cv { namespace triton {
namespace {
// The Python wrappers release the GIL. Serialize metadata changes and dependency
// submission together; device execution remains asynchronous across streams.
std::recursive_mutex& submissionMutex()
{
    static std::recursive_mutex mutex;
    return mutex;
}
typedef std::lock_guard<std::recursive_mutex> Lock;

void check(hipError_t status, const char* what)
{
    if (status != hipSuccess)
        CV_Error(Error::GpuApiCallError, cv::format("cv::triton: %s: %s", what, hipGetErrorString(status)));
}

struct DeviceState
{
    int device;
    hipCtx_t context = nullptr;
    explicit DeviceState(int index) : device(index)
    {
        check(hipInit(0), "initialize HIP");
        int count = 0;
        check(hipGetDeviceCount(&count), "query HIP devices");
        if (index < 0 || index >= count)
            CV_Error(Error::StsBadArg, "cv::triton: invalid HIP device index");
        check(hipDevicePrimaryCtxRetain(&context, device), "retain HIP primary context");
    }
    ~DeviceState() { if (context) (void)hipDevicePrimaryCtxRelease(device); }
};

// Keep the selected primary contexts shared by all owning objects. Releasing
// the last image must not tear down a context also used by runtime kernel caches.
std::shared_ptr<DeviceState> deviceState(int device)
{
    Lock lock(submissionMutex());
    static std::map<int, std::shared_ptr<DeviceState> > states;
    const auto found = states.find(device);
    if (found != states.end()) return found->second;
    auto state = std::make_shared<DeviceState>(device);
    states.emplace(device, state);
    return state;
}

struct ContextScope
{
    hipCtx_t previous = nullptr;
    int previousDevice = 0;
    bool active = false;
    explicit ContextScope(const std::shared_ptr<DeviceState>& state, bool throwing = true)
    {
        hipError_t status = hipGetDevice(&previousDevice);
        if (status == hipSuccess) status = hipCtxGetCurrent(&previous);
        if (status == hipSuccess) status = hipCtxPushCurrent(state->context);
        active = status == hipSuccess;
        if (throwing) check(status, "select HIP context");
    }
    ~ContextScope()
    {
        if (active)
        {
            hipCtx_t popped = nullptr;
            (void)hipCtxPopCurrent(&popped);
            // HIP runtime device selection can outlive a driver context pop.
            (void)hipSetDevice(previousDevice);
            (void)hipCtxSetCurrent(previous);
        }
    }
};

struct Event
{
    std::shared_ptr<DeviceState> state;
    hipEvent_t handle = nullptr;
    explicit Event(const std::shared_ptr<DeviceState>& s) : state(s)
    { check(hipEventCreateWithFlags(&handle, hipEventDisableTiming), "create completion event"); }
    ~Event()
    {
        // Event destruction is deferred by HIP while queued waits still refer to it.
        ContextScope scope(state, false);
        if (scope.active)
        {
            (void)hipEventDestroy(handle);
        }
    }
};

struct Allocation
{
    std::shared_ptr<DeviceState> state;
    void* ptr = nullptr;
    size_t pitch = 0, bytes = 0;
    std::shared_ptr<Event> lastUse;
    explicit Allocation(const std::shared_ptr<DeviceState>& s) : state(s) {}
    ~Allocation()
    {
        ContextScope scope(state, false);
        if (scope.active)
        {
            if (lastUse) (void)hipEventSynchronize(lastUse->handle);
            if (ptr) (void)hipFree(ptr);
        }
    }
};

void notCaptured(hipStream_t stream)
{
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    check(hipStreamIsCapturing(stream, &status), "query HIP stream capture");
    if (status != hipStreamCaptureStatusNone)
        CV_Error(Error::StsNotImplemented, "cv::triton: HIP owning images do not support graph capture");
}
} // namespace

struct HipImage::Impl
{
    std::shared_ptr<DeviceState> state;
    std::shared_ptr<Allocation> allocation;
    DeviceImage view;
    explicit Impl(int device) : state(deviceState(device)) {}
};

struct HipStream::Impl
{
    std::shared_ptr<DeviceState> state;
    hipStream_t handle = nullptr;
    explicit Impl(int device) : state(deviceState(device))
    {
        ContextScope scope(state);
        check(hipStreamCreateWithFlags(&handle, hipStreamNonBlocking), "create HIP stream");
    }
    ~Impl()
    {
        ContextScope scope(state, false);
        if (scope.active)
        {
            (void)hipStreamSynchronize(handle);
            (void)hipStreamDestroy(handle);
        }
    }
};

struct HipAccess
{
    static HipImage::Impl& image(const Ptr<HipImage>& src)
    {
        if (!src || !src->impl->allocation)
            CV_Error(Error::StsBadArg, "cv::triton: source HipImage is empty");
        return *src->impl;
    }
    static cv_triton_runtime::DeviceView view(const HipImage::Impl& src)
    {
        ImageView v;
        v.ptr = src.view.ptr; v.rows = src.view.rows; v.cols = src.view.cols;
        v.type = src.view.type; v.step = src.view.step;
        auto out = detail::toDeviceView(v, src.state->device,
            reinterpret_cast<uintptr_t>(src.allocation->ptr), src.allocation->bytes);
        out.ctx.backend = cv_triton_runtime::Backend::HIP;
        out.ctx.native = reinterpret_cast<uintptr_t>(src.state->context);
        return out;
    }
    static hipStream_t stream(const Ptr<HipStream>& stream, int device)
    {
        if (stream && stream->impl->state->device != device)
            CV_Error(Error::StsBadArg, "cv::triton: HipStream and images must be on the same device");
        hipStream_t handle = stream ? stream->impl->handle : nullptr;
        notCaptured(handle);
        return handle;
    }
    static void create(HipImage::Impl& dst, Size size, int type)
    {
        if (dst.allocation && dst.view.rows == size.height && dst.view.cols == size.width && dst.view.type == type)
            return;
        auto allocation = std::make_shared<Allocation>(dst.state);
        check(hipMallocPitch(&allocation->ptr, &allocation->pitch,
              static_cast<size_t>(size.width) * CV_ELEM_SIZE(type), size.height), "allocate HIP image");
        allocation->bytes = allocation->pitch * size.height;
        DeviceImage v;
        v.ptr = reinterpret_cast<uintptr_t>(allocation->ptr);
        v.rows = size.height; v.cols = size.width; v.type = type; v.step = allocation->pitch;
        dst.allocation = allocation;
        dst.view = v;
    }
    static Ptr<HipImage> output(const Ptr<HipImage>& src, Ptr<HipImage> dst,
                                const detail::OutputInfo& info, bool transparent = false)
    {
        if (dst)
        {
            if (src.get() == dst.get())
                CV_Error(Error::StsBadArg, "cv::triton: in-place operation is not supported");
            if (src->impl->state->device != dst->impl->state->device)
                CV_Error(Error::StsBadArg, "cv::triton: source and destination must be on the same HIP device");
            if (dst->impl->allocation && cv_triton_runtime::views_overlap(view(*src->impl), view(*dst->impl)))
                CV_Error(Error::StsBadArg, "cv::triton: source and destination must not overlap");
        }
        if (transparent && (!dst || !dst->impl->allocation ||
            dst->impl->view.rows != info.size.height || dst->impl->view.cols != info.size.width || dst->impl->view.type != info.type))
            CV_Error(Error::StsBadArg, "cv::triton: BORDER_TRANSPARENT requires initialized dst of the output size and type");
        if (!dst) dst = makePtr<HipImage>(src->impl->state->device);
        create(*dst->impl, info.size, info.type);
        return dst;
    }
    template<class Function>
    static void submit(const Ptr<HipImage>& src, const Ptr<HipImage>& dst, hipStream_t stream, Function function)
    {
        const auto a = src->impl->allocation, b = dst->impl->allocation;
        // Create before submission: even a partially submitted failing operation
        // must protect both the input reads and output writes from early release.
        const auto completion = std::make_shared<Event>(src->impl->state);
        if (a->lastUse) check(hipStreamWaitEvent(stream, a->lastUse->handle, 0), "wait for source use");
        if (b != a && b->lastUse) check(hipStreamWaitEvent(stream, b->lastUse->handle, 0), "wait for destination use");
        try { function(); }
        catch (...)
        {
            finish(a, b, completion, stream);
            throw;
        }
        finish(a, b, completion, stream);
    }
    static void finish(const std::shared_ptr<Allocation>& a, const std::shared_ptr<Allocation>& b,
                       const std::shared_ptr<Event>& event, hipStream_t stream)
    {
        const hipError_t status = hipEventRecord(event->handle, stream);
        if (status != hipSuccess)
        {
            // No event can cover the work: complete it before any owner can free it.
            check(hipStreamSynchronize(stream), "synchronize after completion-event failure");
            check(status, "record completion event");
        }
        a->lastUse = event;
        b->lastUse = event;
    }
};

HipImage::HipImage(int device) : impl(std::make_shared<Impl>(device)) {}
HipImage::~HipImage() {}
void HipImage::upload(InputArray src)
{
    const Mat host = src.getMat();
    if (host.empty() || host.dims != 2 || host.channels() > 4 ||
        (host.depth() != CV_8U && host.depth() != CV_16U && host.depth() != CV_32F))
        CV_Error(Error::StsUnsupportedFormat, "cv::triton::HipImage::upload: expected a non-empty 2-D uint8/uint16/float32 image with 1 to 4 channels");
    Lock lock(submissionMutex());
    ContextScope scope(impl->state);
    notCaptured(nullptr);
    HipAccess::create(*impl, host.size(), host.type());
    if (impl->allocation->lastUse)
        check(hipEventSynchronize(impl->allocation->lastUse->handle), "wait before upload");
    check(hipMemcpy2D(reinterpret_cast<void*>(impl->view.ptr), impl->view.step, host.data, host.step,
          host.cols * host.elemSize(), host.rows, hipMemcpyHostToDevice), "upload HIP image");
    check(hipStreamSynchronize(nullptr), "complete HIP upload");
}
void HipImage::download(OutputArray dst) const
{
    Lock lock(submissionMutex());
    if (!impl->allocation) CV_Error(Error::StsBadArg, "cv::triton::HipImage::download: empty image");
    ContextScope scope(impl->state);
    notCaptured(nullptr);
    if (impl->allocation->lastUse)
        check(hipEventSynchronize(impl->allocation->lastUse->handle), "wait before download");
    dst.create(impl->view.rows, impl->view.cols, impl->view.type);
    Mat host = dst.getMat();
    check(hipMemcpy2D(host.data, host.step, reinterpret_cast<void*>(impl->view.ptr), impl->view.step,
          impl->view.cols * host.elemSize(), impl->view.rows, hipMemcpyDeviceToHost), "download HIP image");
    check(hipStreamSynchronize(nullptr), "complete HIP download");
}
Ptr<HipImage> HipImage::roi(Rect rect) const
{
    Lock lock(submissionMutex());
    const DeviceImage& v = impl->view;
    if (!impl->allocation || rect.x < 0 || rect.y < 0 || rect.width <= 0 || rect.height <= 0 ||
        rect.x > v.cols - rect.width || rect.y > v.rows - rect.height)
        CV_Error(Error::StsBadArg, "cv::triton::HipImage::roi: rectangle must be non-empty and inside the image");
    auto result = makePtr<HipImage>(impl->state->device);
    result->impl->allocation = impl->allocation;
    result->impl->view = v;
    result->impl->view.ptr += rect.y * v.step + rect.x * CV_ELEM_SIZE(v.type);
    result->impl->view.rows = rect.height;
    result->impl->view.cols = rect.width;
    return result;
}
bool HipImage::empty() const { Lock lock(submissionMutex()); return !impl->allocation; }
Size HipImage::size() const { Lock lock(submissionMutex()); return Size(impl->view.cols, impl->view.rows); }
int HipImage::type() const { Lock lock(submissionMutex()); return impl->view.type; }
int HipImage::channels() const { Lock lock(submissionMutex()); return CV_MAT_CN(impl->view.type); }
int HipImage::depth() const { Lock lock(submissionMutex()); return CV_MAT_DEPTH(impl->view.type); }
int HipImage::device() const { return impl->state->device; }
size_t HipImage::step() const { Lock lock(submissionMutex()); return impl->view.step; }
HipStream::HipStream(int device) : impl(std::make_shared<Impl>(device)) {}
HipStream::~HipStream() {}
int HipStream::device() const { return impl->state->device; }
void HipStream::waitForCompletion()
{
    Lock lock(submissionMutex()); ContextScope scope(impl->state);
    check(hipStreamSynchronize(impl->handle), "wait for HIP stream");
}
bool HipStream::queryIfComplete() const
{
    Lock lock(submissionMutex()); ContextScope scope(impl->state);
    const hipError_t status = hipStreamQuery(impl->handle);
    if (status == hipErrorNotReady) return false;
    check(status, "query HIP stream");
    return true;
}
void HipStream::waitFor(const Ptr<HipStream>& producer)
{
    Lock lock(submissionMutex()); ContextScope scope(impl->state);
    if (!producer || producer->device() != device())
        CV_Error(Error::StsBadArg, "cv::triton::HipStream::waitFor: producer must be a stream on the same device");
    notCaptured(impl->handle); notCaptured(producer->impl->handle);
    Event event(impl->state);
    check(hipEventRecord(event.handle, producer->impl->handle), "record stream dependency");
    check(hipStreamWaitEvent(impl->handle, event.handle, 0), "wait for producer stream");
}

Ptr<HipImage> resizeHip(const Ptr<HipImage>& src, Size dsize, Ptr<HipImage> dst,
                        double fx, double fy, int interpolation, Ptr<HipStream> stream)
{
    Lock lock(submissionMutex());
    auto& source = HipAccess::image(src);
    ContextScope scope(source.state);
    const hipStream_t handle = HipAccess::stream(stream, source.state->device);
    const auto info = detail::resizeOutput(HipAccess::view(source), dsize, fx, fy, interpolation);
    dst = HipAccess::output(src, dst, info);
    HipAccess::submit(src, dst, handle, [&] {
        resizeDevice(source.view, HipAccess::image(dst).view, dsize, fx, fy, interpolation,
                     source.state->device, reinterpret_cast<uintptr_t>(handle));
    });
    return dst;
}
Ptr<HipImage> cvtColorHip(const Ptr<HipImage>& src, int code, Ptr<HipImage> dst, Ptr<HipStream> stream)
{
    Lock lock(submissionMutex());
    auto& source = HipAccess::image(src);
    ContextScope scope(source.state);
    const hipStream_t handle = HipAccess::stream(stream, source.state->device);
    const auto info = detail::cvtColorOutput(HipAccess::view(source), code);
    dst = HipAccess::output(src, dst, info);
    HipAccess::submit(src, dst, handle, [&] {
        cvtColorDevice(source.view, HipAccess::image(dst).view, code, source.state->device,
                       reinterpret_cast<uintptr_t>(handle));
    });
    return dst;
}
Ptr<HipImage> warpAffineHip(const Ptr<HipImage>& src, InputArray M, Size dsize, Ptr<HipImage> dst,
                            int flags, int borderMode, const Scalar& borderValue, Ptr<HipStream> stream)
{
    Lock lock(submissionMutex());
    auto& source = HipAccess::image(src);
    ContextScope scope(source.state);
    const hipStream_t handle = HipAccess::stream(stream, source.state->device);
    const auto request = detail::warpRequest(HipAccess::view(source), M, flags, borderMode, borderValue);
    const auto info = detail::warpOutput(request, dsize);
    dst = HipAccess::output(src, dst, info, borderMode == BORDER_TRANSPARENT);
    HipAccess::submit(src, dst, handle, [&] {
        warpAffineDevice(source.view, HipAccess::image(dst).view, request.M, flags, borderMode,
                         borderValue, source.state->device, reinterpret_cast<uintptr_t>(handle));
    });
    return dst;
}
}} // namespace cv::triton

#else
namespace cv { namespace triton {
struct HipImage::Impl {};
struct HipStream::Impl {};
namespace {
void noHip() { CV_Error(Error::StsNotImplemented, "cv::triton: HipImage and HipStream require the HIP backend"); }
}
HipImage::HipImage(int) { noHip(); }
HipImage::~HipImage() {}
void HipImage::upload(InputArray) { noHip(); }
void HipImage::download(OutputArray) const { noHip(); }
Ptr<HipImage> HipImage::roi(Rect) const { noHip(); return Ptr<HipImage>(); }
bool HipImage::empty() const { noHip(); return true; }
Size HipImage::size() const { noHip(); return Size(); }
int HipImage::type() const { noHip(); return 0; }
int HipImage::channels() const { noHip(); return 0; }
int HipImage::depth() const { noHip(); return 0; }
int HipImage::device() const { noHip(); return 0; }
size_t HipImage::step() const { noHip(); return 0; }
HipStream::HipStream(int) { noHip(); }
HipStream::~HipStream() {}
int HipStream::device() const { noHip(); return 0; }
void HipStream::waitForCompletion() { noHip(); }
bool HipStream::queryIfComplete() const { noHip(); return false; }
void HipStream::waitFor(const Ptr<HipStream>&) { noHip(); }
Ptr<HipImage> resizeHip(const Ptr<HipImage>&, Size, Ptr<HipImage>, double, double, int, Ptr<HipStream>)
{ noHip(); return Ptr<HipImage>(); }
Ptr<HipImage> cvtColorHip(const Ptr<HipImage>&, int, Ptr<HipImage>, Ptr<HipStream>)
{ noHip(); return Ptr<HipImage>(); }
Ptr<HipImage> warpAffineHip(const Ptr<HipImage>&, InputArray, Size, Ptr<HipImage>, int, int, const Scalar&, Ptr<HipStream>)
{ noHip(); return Ptr<HipImage>(); }
}} // namespace cv::triton
#endif

namespace cv { namespace triton {
Ptr<HipImage> resizeForPython(const Ptr<HipImage>& src, Size dsize, Ptr<HipImage> dst,
                              double fx, double fy, int interpolation, Ptr<HipStream> stream)
{ return resizeHip(src, dsize, dst, fx, fy, interpolation, stream); }
Ptr<HipImage> cvtColorForPython(const Ptr<HipImage>& src, int code, Ptr<HipImage> dst, Ptr<HipStream> stream)
{ return cvtColorHip(src, code, dst, stream); }
Ptr<HipImage> warpAffineForPython(const Ptr<HipImage>& src, InputArray M, Size dsize, Ptr<HipImage> dst,
                                  int flags, int borderMode, const Scalar& value, Ptr<HipStream> stream)
{ return warpAffineHip(src, M, dsize, dst, flags, borderMode, value, stream); }
}} // namespace cv::triton
