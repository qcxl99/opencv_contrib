// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Standalone CPU control, deliberately outside the native test target's source list.
// Build/run with tools/check_source.py; no OpenCV, GPU SDK or Triton dependency.
#include "compile_protocol.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* what)
{
    if (!value) throw std::runtime_error(what);
}
}

int main(int argc, char** argv) try
{
    using namespace cv_triton_runtime;
    require(argc == 3, "expected Python executable and temporary directory");
    require(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "SHA-256 known-answer vector");
    Json value;
    std::string why;
    require(json_parse("{\"x\":[1,true,\"quote\\\"\"]}", value, why), "valid JSON");
    require(!json_parse("{\"x\":[1,}", value, why), "malformed JSON must fail");

    KernelKey key;
    key.kernel = "test_kernel";
    key.signature = "src:*u8,W:i32";
    key.constexprs = {{"CN", 3}};
    key.cfg = {1, 256, 4, 3, 0};
    key.toolchain.source_sha256 = sha256_hex("source");
    const std::string text = key.text();
    const std::string request = make_request(key, "/source.py", argv[2]);
    require(json_parse(request, value, why), "request JSON");
    std::string rebuilt;
    require(value.get("key") && key_text_from_json(*value.get("key"), rebuilt, why) && rebuilt == text,
            "C++ key/request round trip");
    KernelKey changed = key;
    changed.toolchain.source_sha256 = sha256_hex("new source");
    require(changed.text() != text, "source identity changes key");
    changed = key; changed.target = {"hip", 936, 64};
    require(changed.text() != text, "backend/target changes key");

    Manifest manifest;
    manifest.compiler = "triton-3.6.0";
    manifest.format = "cubin";
    manifest.global_scratch_bytes = manifest.profile_scratch_bytes = 0;
    manifest.abi = {{"src", "*u8", "u64", false},
                    {"global_scratch", "*i8", "u64", true},
                    {"profile_scratch", "*i8", "u64", true}};
    require(launch_abi_ok(manifest, why), "validated CUDA scratch ABI");
    manifest.format = "hsaco";
    require(launch_abi_ok(manifest, why), "validated HIP scratch ABI");
    manifest.global_scratch_bytes = 4;
    require(!launch_abi_ok(manifest, why), "nonzero scratch refused");
    manifest.global_scratch_bytes = 0; manifest.compiler = "unvalidated";
    require(!launch_abi_ok(manifest, why), "unvalidated compiler refused");
    manifest.compiler = "triton-3.6.0"; manifest.abi.back().name = "wrong";
    require(!launch_abi_ok(manifest, why), "unexpected trailing parameter refused");

    std::string dir, status;
    const std::string response = "{\"protocol\":" + json_quote(COMPILE_PROTOCOL) +
        ",\"request_sha256\":" + json_quote(sha256_hex(request)) +
        ",\"status\":\"error\",\"error\":{\"kind\":\"toolchain_unavailable\"}}";
    require(!verify_response(request, response, dir, status, why) && why == "toolchain_unavailable",
            "helper error retains cause");
    require(!verify_response(request + " ", response, dir, status, why) &&
            why == "response answers another request", "response bound to exact request bytes");
    Manifest absent;
    require(!verify_artifact(argv[2], text, absent, why), "incomplete artifact refused");
    HelperCommand command;
    command.python = "/nonexistent/triton-test-python";
    command.helper = "/nonexistent/helper.py";
    command.timeout_s = 1;
    require(!run_request(command, request).status.ok(), "missing toolchain must fail");
    const std::string sleeper = std::string(argv[2]) + "/sleep.py";
    std::ofstream(sleeper) << "import time\ntime.sleep(30)\n";
    command.python = argv[1]; command.helper = sleeper; command.timeout_s = 0.15;
    require(run_request(command, request).kind == "timeout", "hung helper deadline");
    std::cout << "PASS: CPU protocol/key/ABI/error/deadline controls (no GPU or Triton)\n";
    return 0;
}
catch (const std::exception& e)
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
