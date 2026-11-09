"""A fixed three-path rule, not an oracle choosing the best measured case."""

import argparse
import gc
import json
from functools import partial
from types import SimpleNamespace

import torch
from bench_common import assert_equivalent, make_workload
from benchmark_grouped import make_candidates, time_runners
from benchmark_two_stage import run
from selected_plan import allocate_plan


def cases():
    for length in [2048, 8192, 32768, 50000, 65536, 98304, 131072]:
        for counts in [[1], [7], [7] * 4, [7] * 8, [7] * 16, [512]]:
            yield length, counts, None, None
    for length in [50000, 131072]:
        for counts in [[7] * 8, [512]]:
            for overlap in [0.0, 0.5, 1.0]:
                yield length, counts, overlap, None
    yield 131072, [1, 6, 7, 8, 2, 9], None, [2048, 8192, 32768, 50000, 98304, 131072]
    yield 131072, [8] * 8, None, None
    yield 131072, [7] * 32, None, None
    for length, counts in [
        (5120, [7] * 8),
        (16384, [7] * 8),
        (16384, [512]),
        (65537, [512]),
        (81920, [7] * 8),
        (81920, [512]),
        (131072, [2048]),
        (131072, [4096]),
        (262144, [7] * 8),
        (262144, [512]),
    ]:
        yield length, counts, None, None
    for length in [32768, 65536, 81920, 98304]:
        for counts in [[7] * 8, [512]]:
            yield length, counts, 0.0, None
    yield 131072, [2048], 0.0, None
    yield 262144, [512], 0.0, None
    yield 131072, [7] * 16, None, [2048] * 15 + [131072]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["optimized", "reference"], default="optimized")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=1000)
    p.add_argument("--mode", choices=["graph", "eager"], default="graph")
    a = p.parse_args()
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    for idx, (length, counts, overlap, lengths) in enumerate(cases()):
        if not a.start <= idx < a.end:
            continue
        torch.manual_seed(87)
        w = make_workload(length, counts, padding=3)
        if lengths is not None:
            w.lengths.copy_(torch.tensor(lengths, device=w.query.device, dtype=torch.int32))
            w.refresh_metadata()
        minimum = min(lengths) if lengths is not None else length
        fixed = None if overlap is None else make_candidates(w, overlap)[0]
        path, plan = allocate_plan(w, minimum) if a.phase == "optimized" else ("standalone_reference", None)
        print(json.dumps({"event": "start", "case": idx, "phase": a.phase, "path": path}), flush=True)
        baseline = partial(run, w, None, False, fixed)
        runners = {
            "fast_mask_top512": baseline,
            "grouped_total": partial(run, w, plan, True, fixed) if a.phase == "optimized" else baseline,
        }
        timing, outputs, graphs = time_runners(runners, SimpleNamespace(mode=a.mode, warmup=3, samples=5, repeats=5))
        if a.phase == "optimized":
            candidates = w.select(0, 1)[2] if fixed is None else fixed
            offset = 1 if fixed is None else 0
            if offset:
                assert_equivalent(w, outputs["fast_mask_top512"][0], outputs["grouped_total"][0], 0, None)
            for layer in range(1, 5):
                assert_equivalent(
                    w,
                    outputs["fast_mask_top512"][layer - 1 + offset],
                    outputs["grouped_total"][layer - 1 + offset],
                    layer,
                    candidates,
                )
            del candidates
        print(
            json.dumps(
                {
                    "case": idx,
                    "length": length,
                    "counts": counts,
                    "total_queries": sum(counts),
                    "lengths": lengths,
                    "minimum_context": minimum,
                    "controlled_overlap": overlap,
                    "scope": "source_plus_four_consumers_plus_index_preparation"
                    if fixed is None
                    else "four_consumers_plus_preparation",
                    "mode": a.mode,
                    "phase": a.phase,
                    "path": path,
                    "correctness": "pass" if a.phase == "optimized" else "reference",
                    **timing,
                }
            ),
            flush=True,
        )
        del w, plan, fixed, outputs, graphs, runners, baseline
        gc.collect()


if __name__ == "__main__":
    main()
