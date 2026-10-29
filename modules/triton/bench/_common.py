"""Shared setup for the per-mode/per-interface benchmark scripts in this
directory. Not a test module itself -- every bench_*.py here imports this
for the repo-path/module-loading boilerplate every one of them previously
duplicated (bench_full_resize.py, bench_p2_resize.py, bench_real_protocol.py
all re-did this inline before the split).

PLATFORM: CUDA only. These scripts exercise kernelforge-lab's CUDA Triton
kernel (operators/resize/kernel.py) against cv::cuda::resize
(operators/resize/oracle_cuda.py) directly through kernelforge-lab's own
ctypes/AOT layer -- they do not go through this module's cv::triton C++ API
(triton.hpp/core.hpp) at all, and have no HIP/BW1000 counterpart: nothing
here runs under TRITON_BACKEND_HIP. They are carried into this fork as
reference/companion material from the sibling kernelforge-lab project, not
as part of this module's own CMake-driven test/perf suite (see
test/test_resize.cpp and perf/perf_resize.cpp for that, CUDA-only as well
today). Running them requires a kernelforge-lab checkout -- see REPO below.

kernel.py is loaded via importlib.util.spec_from_file_location, not a plain
import, because it lives outside any installed package (operators/resize is
not on sys.path by default) -- same loading convention every existing
bench/test script in the kernelforge-lab repo already uses.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# Default assumes this file sits two levels inside a kernelforge-lab-shaped
# checkout (<repo>/bench/_common.py), which is true in kernelforge-lab
# itself but NOT here (this copy lives at
# modules/triton/bench/_common.py, inside the opencv_contrib tree, not
# inside kernelforge-lab). Point KERNELFORGE_LAB_ROOT at a kernelforge-lab
# checkout to run these from here; the bare default is kept so this file
# stays otherwise identical to the one in kernelforge-lab's own bench/.
_override = os.environ.get("KERNELFORGE_LAB_ROOT")
REPO = Path(_override).resolve() if _override else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "operators" / "resize"))

from kernelforge import benchmark as B  # noqa: E402
from kernelforge import correctness as C  # noqa: E402

CASES_PATH = str(REPO / "operators" / "resize" / "cases.yaml")

_spec = importlib.util.spec_from_file_location(
    "k", str(REPO / "operators" / "resize" / "kernel.py"))
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)

from oracle_cuda import resize_cuda_oracle  # noqa: E402


def candidate_fn(src, dst_w, dst_h, mode, stream):
    K.resize(src, (dst_w, dst_h), mode, stream=stream)


def baseline_fn(src, dst_w, dst_h, mode, stream):
    resize_cuda_oracle(src, dst_w, dst_h, mode, stream=stream)


def cases_for_mode(suite: str, mode: str) -> list[dict]:
    """Cases from the named suite, filtered to one mode.

    `suite` is one of cases.yaml's own suite names (core/performance/
    matrix/qualification/extended/streaming) -- never a hand-rolled list;
    that hand-rolling (bench_p2_resize.py's AREA_CASES/CUBIC_CASES) is
    exactly what this split retires.
    """
    return [c for c in C.load_cases(CASES_PATH, suite) if c["mode"] == mode]


def print_correctness_result(mode: str, res) -> bool:
    ok = res.status == "C1_PASS"
    print(f"  {mode:16s} {res.cases_passed:4d}/{res.cases_run:4d}  "
          f"{'PASS' if ok else 'FAIL -- ' + res.status}")
    for f in res.failures[:10]:
        print(f"      FAIL {f}")
    return ok


def print_performance_result(result: dict) -> None:
    """Print every metric benchmark_case() actually collected, not just
    graph_replay. A case's own CaseTiming.summarise() always carries all
    four auxiliary metrics (host_wall_sync, host_issue_cost, graph_replay,
    cold_start) in `auxiliary`, keyed by kernelforge.benchmark's own
    M_HOST_WALL/M_HOST_ISSUE/M_GRAPH_REPLAY/M_COLD_START constants --
    printing only graph_replay (as this function originally did) threw
    the other three away even though they were already computed, not
    missing.
    """
    for r in result["per_case"]:
        verdict = r["verdict"]
        if verdict == B.MEASUREMENT_INVALID:
            tag = f"INVALID ({r.get('invalid_reason')}, see {r.get('use_instead')})"
        elif r.get("speedup_vs_native") is not None:
            tag = f"{r['verdict']:12s} speedup={r['speedup_vs_native']:.3f}x"
        else:
            tag = r["verdict"]
        print(f"  {r['case_id']:42s} {tag}")

        aux = r.get("auxiliary", {})
        parts = []

        wall = aux.get(B.M_HOST_WALL, {})
        if wall.get("candidate_ms") is not None:
            if wall.get("baseline_ms"):
                parts.append(f"host_wall_sync: cand={wall['candidate_ms']:.4f}ms "
                             f"base={wall['baseline_ms']:.4f}ms "
                             f"speedup={wall['baseline_ms'] / wall['candidate_ms']:.3f}x")
            else:
                parts.append(f"host_wall_sync: cand={wall['candidate_ms']:.4f}ms (no baseline)")

        issue = aux.get(B.M_HOST_ISSUE, {})
        if issue.get("candidate_ms") is not None:
            if issue.get("baseline_ms"):
                parts.append(f"host_issue_cost: cand={issue['candidate_ms']:.4f}ms "
                             f"base={issue['baseline_ms']:.4f}ms")
            else:
                parts.append(f"host_issue_cost: cand={issue['candidate_ms']:.4f}ms (no baseline)")

        graph = aux.get(B.M_GRAPH_REPLAY, {})
        if graph.get("speedup") is not None:
            parts.append(f"graph_replay: cand={graph['candidate_ms']:.4f}ms "
                         f"base={graph['baseline_ms']:.4f}ms speedup={graph['speedup']:.3f}x")
        elif graph.get("candidate_ms") is not None:
            parts.append(f"graph_replay: cand={graph['candidate_ms']:.4f}ms "
                         f"({graph.get('note', 'not symmetric')})")

        cold = aux.get(B.M_COLD_START, {})
        if cold.get("candidate_ms") is not None:
            parts.append(f"cold_start: cand={cold['candidate_ms']:.4f}ms")

        for p in parts:
            print(f"      {p}")

        if r.get("launch_gap_sides"):
            print(f"      launch_gap_sides: {r['launch_gap_sides']}")

    print(f"\nSummary: {result.get('summary', {})}")


def paired_alternating_graph_replay(fn_a, fn_b, stream, block_iters: int,
                                    graph_batch: int, blocks: int = 7):
    """Median speedup of fn_b/fn_a over `blocks` alternating-order repeats.

    batch's resize_batch() vs a loop-of-singles has no cases.yaml entry
    (different calling convention -- resize_batch is not one of the
    per-image Cases every other bench script here measures), so it
    cannot go through benchmark_case()/run_benchmark() directly. This
    applies the SAME two disciplines by hand instead of falling back to
    a plain sequential measurement (the exact simplification
    bench_p2_resize.py made and this split retires): alternate which
    side runs first across blocks, and take the median across blocks
    rather than a single reading or an arithmetic mean (a single slow
    block from transient system noise would skew a mean; the median is
    immune to that).
    """
    ratios = []
    for b in range(blocks):
        a_first = (b % 2 == 0)
        if a_first:
            ta = B._graph_replay_ms(fn_a, stream, block_iters, graph_batch)
            tb = B._graph_replay_ms(fn_b, stream, block_iters, graph_batch)
        else:
            tb = B._graph_replay_ms(fn_b, stream, block_iters, graph_batch)
            ta = B._graph_replay_ms(fn_a, stream, block_iters, graph_batch)
        if ta and tb:
            ratios.append(tb / ta)
    if not ratios:
        return None, None, None
    ratios.sort()
    lo, hi, med = ratios[0], ratios[-1], ratios[len(ratios) // 2]
    return lo, hi, med
