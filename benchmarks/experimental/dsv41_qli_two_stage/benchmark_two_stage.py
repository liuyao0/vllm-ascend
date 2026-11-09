"""Full source chain and controlled-overlap consumer chain; includes planning/packing."""

import argparse
import gc
import json
import statistics
from functools import partial
from types import SimpleNamespace

import torch
from bench_common import assert_equivalent, make_workload, prepare_indexer_indices
from benchmark_grouped import make_candidates, time_runners
from packed_plan import PackedPlan


def finish(work, output):
    return prepare_indexer_indices(
        output.squeeze(1),
        work.positions,
        1,
        lengths_output=torch.empty((output.shape[0], 1), dtype=torch.int32, device=output.device),
    ).unsqueeze(1)


def run(work, packed, optimized, fixed_candidates=None):
    outputs = []
    if fixed_candidates is None:
        source, _, candidates = work.select(0, 9 if optimized else 1)
        outputs.append(source)
    else:
        candidates = fixed_candidates
    if optimized:
        packed.prepare(candidates)
    for layer in range(1, 5):
        outputs.append(packed.select(layer)[0] if optimized else work.select(layer, 2, candidates)[0])
    return tuple(finish(work, output) for output in outputs)


def cases(suite):
    if suite == "dspark":
        return [(length, [7] * requests, None) for length in [50000, 131072] for requests in [2, 4, 16]] + [
            (131072, [8] * 8, None),
            (131072, [7] * 32, None),
        ]
    if suite == "source":
        return [
            (2048, [7] * 2, None),
            (8192, [7] * 8, None),
            (32768, [7] * 8, None),
            (50000, [7], None),
            (50000, [7] * 8, None),
            (50000, [512], None),
            (131072, [7], None),
            (131072, [7] * 8, None),
            (131072, [512], None),
            (131072, [2048], None),
        ]
    return (
        [(length, [7] * 8, p) for length in [50000, 131072] for p in [0.0, 0.25, 1.0]]
        + [(131072, [512], p) for p in [0.0, 0.25, 0.5, 1.0]]
        + [(131072, [2048], 0.0), (131072, [1, 6, 7, 8, 2, 9], 0.0)]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["source", "controlled", "dspark"], default="source")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--mode", choices=["eager", "graph"], default="graph")
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    timing_args = SimpleNamespace(mode=args.mode, warmup=3, samples=3, repeats=3)
    for idx, (length, counts, overlap) in enumerate(cases(args.suite)):
        if not args.start <= idx < args.end:
            continue
        case = {
            "case": idx,
            "suite": args.suite,
            "length": length,
            "requests": len(counts),
            "counts": counts,
            "total_queries": sum(counts),
            "mode": args.mode,
            "scope": "source_plus_four_consumers_plus_index_preparation"
            if overlap is None
            else "four_consumers_plus_plan_pack_and_index_preparation",
        }
        print(json.dumps({"event": "start", **case}), flush=True)
        torch.manual_seed(83)
        work = make_workload(length, counts, padding=3 if len(counts) > 1 else 0)
        packed = PackedPlan.allocate(work)
        if overlap is None:
            candidates = work.select(0, 1)[2]
            fixed = None
            sets = [set(row[row >= 0].tolist()) for row in candidates.cpu().reshape(-1, 2048)]
            pairs = []
            start = 0
            for count in counts:
                for first in range(start, start + count, 4):
                    for i in range(first, min(first + 4, start + count)):
                        for j in range(first, i):
                            pairs.append(len(sets[i] & sets[j]) / len(sets[i]))
                start += count
            stats = {"mean_pair_overlap": statistics.mean(pairs), "candidates": "native_source_from_synthetic_q_k"}
        else:
            candidates, stats = make_candidates(work, overlap, holes=idx == 11)
            fixed = candidates
        timing, outputs, graphs = time_runners(
            {
                "fast_mask_top512": partial(run, work, packed, False, fixed),
                "grouped_total": partial(run, work, packed, True, fixed),
            },
            timing_args,
        )
        offset = 1 if overlap is None else 0
        if offset:
            torch.testing.assert_close(outputs["fast_mask_top512"][0], outputs["grouped_total"][0])
        for layer in range(1, 5):
            assert_equivalent(
                work,
                outputs["fast_mask_top512"][layer - 1 + offset],
                outputs["grouped_total"][layer - 1 + offset],
                layer,
                candidates,
            )
        for k in ["samples", "device_us"]:
            timing[k]["two_stage_total"] = timing[k].pop("grouped_total")
        timing["two_stage_speedup"] = timing.pop("grouped_total_speedup")
        print(
            json.dumps({**case, **stats, "correctness": "pass", "workspace_bytes": packed.workspace_bytes, **timing}),
            flush=True,
        )
        del work, packed, candidates, fixed, outputs, graphs
        gc.collect()


if __name__ == "__main__":
    main()
