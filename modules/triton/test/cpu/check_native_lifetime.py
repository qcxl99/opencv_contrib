#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright 2026, The FlagOS Contributors.
"""Compile selected backend lifetime/context functions against injected CPU driver calls.

Reads the module sources and uses a temporary C++ harness to check completion,
cleanup failures and primary-context ownership. No SDK or GPU fault is needed.
Prints control results and exits nonzero on a failed compile/assertion; this does
not establish real-driver or hardware qualification.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
import tempfile

PRELUDE = r'''
#include "compile_protocol.hpp"
#include "lifetime.hpp"
#include <atomic>
#include <array>
#include <functional>
#include <cassert>
#include <iostream>
#include <thread>
#include <memory>
#include <list>
using namespace cv_triton_runtime;
using NativeContext = void*;
thread_local NativeContext activeContext = nullptr;
std::atomic<int> retains{0}, sets{0}, failSets{0}, idQueries{0}, invalidId{0};
bool nativeId = false;
int init(unsigned) { return 0; }
int getCurrent(NativeContext* c) { *c = activeContext; return 0; }
int setCurrent(NativeContext c) { ++sets; if (failSets > 0) { --failSets; return 5; } activeContext = c; return 0; }
int deviceGet(int* d, int n) { *d = n; return 0; }
int getDevice(int* d) { *d = activeContext == reinterpret_cast<void*>(0x2000) ? 1 : 0; return 0; }
int retain(NativeContext* c, int d) { ++retains; *c = reinterpret_cast<void*>(0x1000 + d * 0x1000); return 0; }
int getId(NativeContext c, unsigned long long* g) { ++idQueries; if (invalidId) return invalidId == 1 ? 2 : 5; *g = reinterpret_cast<std::uintptr_t>(c); return 0; }
int getState(int, unsigned* flags, int* active) { *flags = 0; *active = !invalidId; return invalidId == 2 ? 5 : 0; }
void* fakeDlsym(void*, const char*) { return nativeId ? reinterpret_cast<void*>(&getId) : nullptr; }
#define dlsym fakeDlsym
#define RTLD_DEFAULT nullptr
bool driverReady() { return true; }
Status error(Reason r, const std::string& m) { Status s; s.code = Code::Error; s.reason = r; s.message = m; return s; }
std::string driverError(int r) { return std::to_string(r); }
'''
ALIASES = r'''
using CUcontext = NativeContext; using CUresult = int; using CUdevice = int;
constexpr int CUDA_SUCCESS = 0, CUDA_ERROR_INVALID_CONTEXT = 2, CUDA_ERROR_CONTEXT_IS_DESTROYED = 3,
              CUDA_ERROR_DEINITIALIZED = 4, CUDA_ERROR_NOT_READY = 7;
#define cuCtxGetCurrent getCurrent
#define cuCtxSetCurrent setCurrent
#define cuCtxGetDevice getDevice
#define cuCtxGetId getId
#define cuDeviceGet deviceGet
#define cuDevicePrimaryCtxRetain retain
#define cuErr driverError
using hipCtx_t = NativeContext; using hipError_t = int; using hipDevice_t = int;
constexpr int hipSuccess=0, hipErrorInvalidContext=2, hipErrorContextIsDestroyed=3,
              hipErrorDeinitialized=4, hipErrorInvalidValue=6, hipErrorNotReady=7;
#define hipCtxGetCurrent getCurrent
#define hipCtxSetCurrent setCurrent
#define hipCtxGetDevice getDevice
#define hipDeviceGet deviceGet
#define hipDevicePrimaryCtxRetain retain
#define hipDevicePrimaryCtxGetState getState
#define hipErr driverError
'''

TARGET_PRELUDE = r'''
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR=0,
              CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR=1, CU_DEVICE_ATTRIBUTE_WARP_SIZE=2;
int attributeQueries=0, attributeFailureAt=0;
int getAttribute(int* value, int attribute, int device) {
    ++attributeQueries;
    if (attributeQueries==attributeFailureAt) return 5;
    *value=attribute==0 ? 9+device : attribute==1 ? 0 : 32;
    return 0;
}
#define cuDeviceGetAttribute getAttribute
'''
TARGET_TEST = r'''
void exerciseTargets() {
    int major=0,minor=0,warp=0;
    assert(deviceAttributes(0,major,minor,warp).ok());
    assert(major==9 && minor==0 && warp==32 && attributeQueries==3);
    assert(deviceAttributes(1,major,minor,warp).ok());
    assert(major==10 && minor==0 && warp==32 && attributeQueries==6);
    for(int fail=1;fail<=3;++fail) {
        major=71;minor=72;warp=73;
        attributeFailureAt=attributeQueries+fail;
        assert(!deviceAttributes(0,major,minor,warp).ok());
        assert(major==71 && minor==72 && warp==73); // no partial target publication
        attributeFailureAt=0;
        assert(deviceAttributes(0,major,minor,warp).ok());
        assert(major==9 && minor==0 && warp==32); // every failure is retryable
    }
}
'''

REAPER_PRELUDE = r'''
using CUevent = void*; using hipEvent_t = void*;
using CUstream = void*; using hipStream_t = void*;
using CUmodule = void*; using CUfunction = void*; using CUdeviceptr = std::uintptr_t;
using hipModule_t = void*; using hipFunction_t = void*;
std::atomic<long long> g_module_frees{0}, g_table_frees{0};
int freeStatus = 0, waitStatus = 0, freeCalls = 0, unloadCalls = 0;
int freeMemory(std::uintptr_t) { ++freeCalls; return freeStatus; }
int freeMemory(void*) { ++freeCalls; return freeStatus; }
int freeAsync(std::uintptr_t p, void*) { return freeMemory(p); }
int freeAsync(void* p, void*) { return freeMemory(p); }
int unloadModule(void*) { ++unloadCalls; return freeStatus; }
int waitEvent(void*, void*, unsigned) { return waitStatus; }
#define cuMemFree freeMemory
#define hipFree freeMemory
#define cuMemFreeAsync freeAsync
#define hipFreeAsync freeAsync
#define cuModuleUnload unloadModule
#define hipModuleUnload unloadModule
#define cuStreamWaitEvent waitEvent
#define hipStreamWaitEvent waitEvent
int launchCalls = 0;
std::function<void(void**)> inspectArguments;
int fakeLaunch(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
               unsigned, void*, void** params, void**) {
    ++launchCalls; inspectArguments(params); return 0;
}
#define cuLaunchKernel fakeLaunch
#define hipModuleLaunchKernel fakeLaunch
struct CtxGuard { bool valid = true; explicit CtxGuard(NativeContext) {} };
std::atomic<int> eventQueries{0}, eventDestroys{0}, diagnostics{0};
Completion completion = Completion::NotReady;
Completion eventCompletion(void*) { ++eventQueries; return completion; }
int uploadQueries = 0;
int queryUpload(void*) {
    ++uploadQueries;
    return completion == Completion::Complete ? 0 : completion == Completion::NotReady ? 7 : 5;
}
#define cuEventQuery queryUpload
#define hipEventQuery queryUpload
void cleanupError(const std::string&) { ++diagnostics; }
int destroyStatus=0;
int destroyEvent(void*) { ++eventDestroys; return destroyStatus; }
#define cuEventDestroy destroyEvent
#define hipEventDestroy destroyEvent
struct State { std::list<Pending> pending, failed_events; struct { long long pending = 0; } st; };
'''
REAPER_TEST = r'''
void exerciseReap(NativeContext context, std::uint64_t generation) {
    State state; auto owner=std::make_shared<int>(7);
    state.pending.push_back({context,generation,reinterpret_cast<void*>(9),false,{owner}});
    std::vector<std::shared_ptr<void>> released;
    assert(reapLocked(state,released,true).ok() && eventQueries==0 && state.pending.size()==1);
    invalidId=2; const int unknownQueries=eventQueries;
    assert(!reapLocked(state,released).ok() && state.pending.size()==1 && eventQueries==unknownQueries);
    invalidId=0; diagnostics=0; state.pending.front().query_error_reported=false;
    completion=Completion::NotReady;
    assert(reapLocked(state,released).ok() && state.pending.size()==1 && released.empty());
    completion=Completion::Error;
    assert(!reapLocked(state,released).ok() && state.pending.size()==1 && released.empty() && diagnostics==1);
    assert(!reapLocked(state,released).ok() && diagnostics==1); // observable, not log spam
    completion=Completion::Complete;
    assert(reapLocked(state,released).ok() && state.pending.empty() && released.size()==1 && eventDestroys==1);
    assert(reapLocked(state,released).ok() && eventDestroys==1);
    state.pending.push_back({context,generation,reinterpret_cast<void*>(9),false,{owner}});
    destroyStatus=7;
    assert(!reapLocked(state,released).ok() && state.pending.empty() && state.failed_events.size()==1);
    const int failedDestroys=eventDestroys;
    assert(reapLocked(state,released).ok() && eventDestroys==failedDestroys && state.failed_events.size()==1);
    invalidId=1;
    assert(reapLocked(state,released).ok() && state.failed_events.empty() && eventDestroys==failedDestroys);
    invalidId=0;destroyStatus=0;

    state.pending.push_back({context,generation,reinterpret_cast<void*>(9),false,{owner}});
    invalidId=1; const int queries=eventQueries, destroys=eventDestroys;
    assert(reapLocked(state,released).ok() && state.pending.empty());
    assert(eventQueries==queries && eventDestroys==destroys); // known destroyed context
    invalidId=0;
    // The eager ABI can retain module plus BOTH table families. Release all three
    // after completion, while their destruction remains outside the state lock.
    released.clear();
    auto table1=std::make_shared<int>(8), table2=std::make_shared<int>(9);
    state.pending.push_back({context,generation,reinterpret_cast<void*>(9),false,{owner,table1,table2}});
    completion=Completion::NotReady;
    assert(reapLocked(state,released).ok() && released.empty());
    assert(owner.use_count()==2 && table1.use_count()==2 && table2.use_count()==2);
    completion=Completion::Complete;
    assert(reapLocked(state,released).ok() && state.pending.empty() && released.size()==3);
    assert(owner.use_count()==2 && table1.use_count()==2 && table2.use_count()==2);
    released.clear();
    assert(owner.use_count()==1 && table1.use_count()==1 && table2.use_count()==1);
}
'''

RESOURCE_TEST = r'''
void exerciseResources(NativeContext context, std::uint64_t generation) {
    TableRes dependency;
    dependency.uploaded=reinterpret_cast<void*>(9);
    dependency.upload_stream=reinterpret_cast<void*>(1);
    bool waited=false;
    Status status;
    completion=Completion::NotReady;waitStatus=0;
    const int queriesBefore=uploadQueries;
    assert(dependency.prepareUploadDependency(reinterpret_cast<void*>(1),true,waited,status));
    assert(!waited && uploadQueries==queriesBefore); // capture never queries an event
    assert(dependency.prepareUploadDependency(reinterpret_cast<void*>(2),false,waited,status));
    assert(waited && !dependency.upload_complete); // pending cross-stream dependency remains
    completion=Completion::Complete;
    assert(dependency.prepareUploadDependency(reinterpret_cast<void*>(2),false,waited,status));
    assert(!waited && dependency.upload_complete);
    const int completedQueries=uploadQueries;
    assert(dependency.prepareUploadDependency(reinterpret_cast<void*>(3),true,waited,status));
    assert(!waited && uploadQueries==completedQueries);
    TableRes failedDependency;failedDependency.uploaded=reinterpret_cast<void*>(9);
    completion=Completion::Error;
    assert(!failedDependency.prepareUploadDependency(reinterpret_cast<void*>(2),false,waited,status));
    assert(!status.ok() && !failedDependency.upload_complete);
    Retirement<TableRes> tables;
    auto* table=new TableRes();table->ctx=context;table->gen=generation;
    table->base=256;table->uploaded=reinterpret_cast<void*>(9);
    table->upload_started=table->upload_recorded=true;table->host.resize(64,17);
    completion=Completion::NotReady;waitStatus=5;
    tables.dispose(table);
    assert(tables.size()==1 && table->host[0]==17 && freeCalls==0);
    completion=Completion::Error;tables.retry(false);
    assert(tables.size()==1 && table->host.size()==64 && freeCalls==0);
    completion=Completion::Complete;waitStatus=0;freeStatus=7;
    tables.retry(false);
    assert(tables.size()==1 && freeCalls==1 && table->cleanup_failed && tables.hasFailure());
    tables.retry(false);
    assert(freeCalls==1); // unknown irreversible outcome must not double-free
    invalidId=1;tables.retry(false);invalidId=0;
    assert(tables.size()==0 && freeCalls==1 && !tables.hasFailure());
    Retirement<ModuleRes> modules;
    auto* module=new ModuleRes();module->ctx=context;module->gen=generation;
    module->mod=reinterpret_cast<void*>(8);
    modules.dispose(module);
    assert(modules.size()==1 && unloadCalls==1);
    modules.retry(false);assert(unloadCalls==1);
    invalidId=1;modules.retry(false);invalidId=0;
    assert(modules.size()==0 && unloadCalls==1);
    freeStatus=0;
}
'''

ARGUMENT_TEST = r'''
void exerciseArguments() {
    ModuleRes module;
    module.man.abi = {{"src","*u8","u64",false}, {"count","i32","u32",false},
                      {"scale","fp32","f32",false}, {"scratch","*u8","u64",true}};
    module.man.threads_per_block=128;
    LaunchArguments args;
    Arg p;p.kind='p';p.p=0x1234;args["src"]=p;
    Arg n;n.kind='i';n.i=-7;args["count"]=n;n.i=13;args["count"]=n;
    Arg f;f.kind='f';f.f=1.25f;args["scale"]=f;
    inspectArguments=[](void** params){
        assert(*static_cast<std::uintptr_t*>(params[0])==0x1234);
        assert(*static_cast<int*>(params[1])==13);
        assert(*static_cast<float*>(params[2])==1.25f);
        assert(*static_cast<std::uintptr_t*>(params[3])==0);
    };
    unsigned grid[3]={1,1,1};
    assert(launch(module,args,grid,nullptr).ok() && launchCalls==1);
    module.man.abi[1].name="missing";
    assert(!launch(module,args,grid,nullptr).ok() && launchCalls==1);
    module.man.abi[1].name="count";module.man.abi[1].type="fp32";
    assert(!launch(module,args,grid,nullptr).ok() && launchCalls==1);
    LaunchArguments large;module.man.abi.clear();
    for(int i=0;i<40;++i){
        const std::string name="v"+std::to_string(i);
        n.i=i*3;large[name]=n;
        module.man.abi.push_back({name,"i32","u32",false});
    }
    inspectArguments=[](void** params){for(int i=0;i<40;++i)assert(*static_cast<int*>(params[i])==i*3);};
    assert(launch(module,large,grid,nullptr).ok() && launchCalls==2);
    inspectArguments={};
}
'''

MAIN = r'''
int main(int argc, char** argv) {
    nativeId = argc > 1 && std::string(argv[1]) == "id";
    NativeContext context = nullptr; std::uint64_t generation = 0;
    failSets = 1;
    assert(!currentContext(0, context, generation).ok());
    assert(retains == 1); // failed SetCurrent did not leak an additional thread retain
    assert(currentContext(0, context, generation).ok());
    assert(retains == 1 && context == reinterpret_cast<void*>(0x1000));
    const auto first = generation;
    exerciseReap(context,generation);
    exerciseResources(context,generation);
    exerciseArguments();
    for (int i = 0; i < 8; ++i) assert(currentContext(0, context, generation).ok() && generation == first);
    std::vector<std::thread> threads;
    for (int i = 0; i < 64; ++i) threads.emplace_back([&]() {
        NativeContext c=nullptr; std::uint64_t g=0;
        assert(currentContext(0,c,g).ok() && g==first);
        activeContext=nullptr; // host thread exit does not drop runtime's cache owner
    });
    for(auto& t:threads)t.join();
    assert(retains == 1);
    activeContext=nullptr;
    assert(currentContext(1,context,generation).ok());
    assert(retains == 2 && generation != first);
    invalidId = 1;
    assert(contextLife(context,generation)==ContextLife::Gone);
    invalidId = 0;
    std::cout << "PASS: actual context acquisition + lifetime, 64 threads, SetCurrent failure, two devices, native-ID=" << nativeId << ", retains=" << retains << "\n";
}
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='c++')
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parents[2] / 'src' / 'runtime'
    with tempfile.TemporaryDirectory(prefix='opencv-triton-native-lifetime-') as directory:
        for backend in ('cuda', 'hip'):
            text = (runtime / (backend + '_runtime.cpp')).read_text()
            start = text.index('ContextLife contextLife') if backend == 'cuda' else text.index('hipError_t primaryContext(int device, hipCtx_t& context);')
            end = text.index('struct CleanupDiagnostics', start)
            contexts = text[start:end]
            start = text.index('Status currentContext(')
            end = text.index('// ---------------------------------------------------------------- modules', start)
            reap_start = text.index("Status reapLocked(")
            reap_end = text.index("Status currentContext(", reap_start)
            resource_start = text.index("struct ModuleRes")
            resource_end = text.index("thread_local bool recordingCapture", resource_start)
            argument_start = text.index("struct Arg\n")
            argument_end = text.index("// The div tables", argument_start)
            code = PRELUDE + ALIASES + TARGET_PRELUDE + contexts + text[start:end] + REAPER_PRELUDE.replace("struct State", text[text.index("struct Pending\n"):text.index("template <class V> struct Lru")] + "struct State") + text[resource_start:resource_end] + text[reap_start:reap_end] + text[argument_start:argument_end] + REAPER_TEST + RESOURCE_TEST + ARGUMENT_TEST + (TARGET_TEST if backend == "cuda" else "") + MAIN.replace("    exerciseArguments();", "    exerciseArguments();\n    exerciseTargets();" if backend == "cuda" else "    exerciseArguments();")
            path = Path(directory) / (backend + '.cpp'); path.write_text(code)
            executable = path.with_suffix('')
            subprocess.run(shlex.split(args.cxx) + ['-std=c++17', '-pthread', '-O2', '-Wall', '-Wextra',
                           '-I' + str(runtime), str(path), '-o', str(executable)], check=True)
            for mode in ('id', 'fallback') if backend == 'hip' else ('id',):
                subprocess.run([str(executable), mode], check=True)

if __name__ == '__main__':
    main()
