"""Four-query union microbench; all totals include device plan generation."""

import argparse
import gc
import itertools
import json
import statistics
from functools import partial

import torch
from bench_common import assert_equivalent, make_workload, measure
from grouped_plan import CAPACITY, HEADER, GroupedPlan


def make_candidates(work, overlap, holes=False):
    """Shared random subset plus distinct tails where context capacity allows.

    Low-overlap sets are disjoint within each four-query group when possible.
    At short contexts the unavoidable overlap is measured, not labeled zero.
    Both candidate order and physical pages are shuffled.
    """
    starts, lengths = work.starts.cpu().tolist(), work.lengths.cpu().tolist()
    output = torch.full((work.query.shape[1], 1, 2048), -1, dtype=torch.int32)
    stats = []
    for request, length in enumerate(lengths):
        for row in range(starts[request], starts[request + 1], 4):
            nq = min(4, starts[request + 1] - row)
            visible = length - (starts[request + 1] - row) + 1
            available = (visible + 7) // 8
            selected = min(1024 if holes else 2048, available)
            shared = round(selected * overlap)
            pool = torch.randperm(available)
            common, tail = pool[:shared], pool[shared:]
            sets = []
            for q in range(nq):
                n = selected - shared
                extra = tail[(torch.arange(n) + q * n) % max(1, tail.numel())]
                ids = torch.cat((common, extra))
                ids = ids[torch.randperm(ids.numel())].int()
                if holes:
                    output[row + q, 0, : 2 * ids.numel() : 2] = ids
                else:
                    output[row + q, 0, : ids.numel()] = ids
                sets.append(set(ids.tolist()))
            union = set().union(*sets)
            pairs = [len(sets[i] & sets[j]) / selected for i in range(nq) for j in range(i)]
            sorted_union = sorted(union)
            stats.append(
                {
                    "unique_blocks": len(union),
                    "selected_sum": nq * selected,
                    "pair_overlap": sum(pairs) / len(pairs) if pairs else None,
                    "nonadjacent": sum(b != a + 1 for a, b in itertools.pairwise(sorted_union))
                    / max(1, len(sorted_union) - 1),
                }
            )
    pair_stats = [s["pair_overlap"] for s in stats if s["pair_overlap"] is not None]
    return output.to(work.query.device), {
        "requested_overlap": overlap,
        "mean_pair_overlap": statistics.mean(pair_stats) if pair_stats else None,
        "union_read_saving": 1 - sum(s["unique_blocks"] for s in stats) / sum(s["selected_sum"] for s in stats),
        "mean_nonadjacent_gap_fraction": statistics.mean(s["nonadjacent"] for s in stats),
    }


def check_plan(work, candidates, plan):
    rows = candidates.cpu().reshape(-1, 2048)
    data, table = plan.cpu(), work.table.cpu()
    starts, lengths = work.starts.cpu().tolist(), work.lengths.cpu().tolist()
    seen = []
    actual_groups = sum((n + 3) // 4 for n in work.counts)
    assert bool((data[:, 6] == actual_groups).all())
    assert bool((data[:actual_groups, 1] > 0).all())
    assert bool((data[actual_groups:, 1] == 0).all())
    for record in data:
        row, nq, request, visible, count = record[:5].tolist()
        if nq == 0:
            assert count == 0
            continue
        assert row >= starts[request] and row + nq <= starts[request + 1]
        assert visible == lengths[request] - starts[request + 1] + row + 1
        sets = [set(rows[row + q][rows[row + q] >= 0].tolist()) for q in range(nq)]
        wanted = set(range(min(64, (visible + nq - 1 + 7) // 8))).union(*sets)
        origins = record[HEADER : HEADER + count] // 8
        live = origins >= 0
        assert sorted(origins[live].tolist()) == sorted(wanted)
        assert origins[live].unique().numel() == len(wanted)
        masks = record[HEADER + CAPACITY : HEADER + CAPACITY + count][live].tolist()
        expected_masks = [sum((1 << q) for q in range(nq) if block in sets[q]) for block in origins[live].tolist()]
        for idx, block in enumerate(origins[live].tolist()):
            if block < 64:
                assert masks[idx] == expected_masks[idx]
            else:
                assert masks[idx] == 2139095040
        physical = record[HEADER + 2 * CAPACITY : HEADER + 2 * CAPACITY + count][live]
        ids = origins[live].long()
        assert torch.equal(physical, table[request, ids // 16] * 128 + ids % 16 * 8)
        for bucket in range(16):
            base, size = record[8 + 2 * bucket : 10 + 2 * bucket].tolist()
            ids = record[HEADER + base : HEADER + base + size] // 8
            masks = record[HEADER + CAPACITY + base : HEADER + CAPACITY + base + size]
            if bucket == 0:
                assert bool(((ids < 64) | (ids == -1)).all())
            else:
                assert bool((masks[ids >= 0] == 2139095040).all())
                for block in ids[ids >= 0].tolist():
                    assert sum(1 << q for q in range(nq) if block in sets[q]) == bucket
                assert bool((ids[ids >= 0] >= 64).all())
        seen.extend(range(row, row + nq))
    assert sorted(seen) == list(range(sum(work.counts)))


def cases(suite):
    if suite == "smoke":
        return [
            (32768, [8], 0.0, False),
            (131072, [64], 0.5, False),
            (511, [7], 0.0, True),
            (131072, [1, 6, 7, 8, 2, 9], 0.0, True),
        ]
    if suite == "main":
        return [
            (length, [q], p, False)
            for length in (8192, 32768, 50000, 131072)
            for q in (64, 512, 2048)
            for p in (0.0, 0.5, 1.0)
        ]
    if suite == "boundary":
        return (
            [(131072, [q], p, False) for q in (7, 32, 128, 4096) for p in (0.0, 0.25, 0.8)]
            + [(131072, [7] * 8, p, False) for p in (0.0, 0.5, 1.0)]
            + [(32768, [3, 7, 129], p, True) for p in (0.0, 0.5, 1.0)]
        )
    return (
        [(131072, [q], p, False) for q in (64, 512, 2048) for p in (0.0, 0.5, 1.0)]
        + [(131072, [6, 7, 8, 6, 7, 8], p, False) for p in (0.0, 0.5, 1.0)]
        + [(131072, [1, 6, 7, 8, 2, 9], 0.0, True)]
    )


def time_runners(runners, args):
    graphs, outputs = [], {}
    for name, run in list(runners.items()):
        for _ in range(args.warmup):
            outputs[name] = run()
        torch.npu.synchronize()
        if args.mode == "graph":
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs[name] = run()
            graphs.append(graph)
            runners[name] = graph.replay
    if args.mode == "graph":
        for run in runners.values():
            for _ in range(args.warmup):
                run()
        torch.npu.synchronize()
    samples = {name: [] for name in runners}
    for sample in range(args.samples):
        order = list(runners)
        if sample % 2:
            order.reverse()
        for name in order:
            samples[name].append(measure(runners[name], args.repeats))
    medians = {name: statistics.median(s["device_us"] for s in records) for name, records in samples.items()}
    return (
        {
            "samples": samples,
            "device_us": medians,
            "grouped_total_speedup": medians["fast_mask_top512"] / medians["grouped_total"],
        },
        outputs,
        graphs,
    )


def four_baseline(work, candidates):
    return tuple(work.select(layer, 2, candidates) for layer in range(1, 5))


def four_grouped(plan, candidates):
    plan.prepare(candidates)
    return tuple(plan.select(layer) for layer in range(1, 5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["smoke", "main", "boundary", "graph"], default="smoke")
    parser.add_argument("--mode", choices=["eager", "graph"], default="eager")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10000)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    print(
        json.dumps(
            {
                "device": torch.npu.get_device_name(),
                "torch": torch.__version__,
                "args": vars(args),
            }
        ),
        flush=True,
    )
    for index, (length, counts, overlap, holes) in enumerate(cases(args.suite)):
        if not args.start <= index < args.end:
            continue
        case = {
            "case": index,
            "length": length,
            "counts": counts,
            "queries": sum(counts),
            "overlap": overlap,
            "holes": holes,
            "mode": args.mode,
        }
        print(json.dumps({"event": "start", **case}), flush=True)
        torch.manual_seed(83)
        work = make_workload(length, counts, padding=3 if len(counts) > 1 else 0)
        candidates, stats = make_candidates(work, overlap, holes)
        plan = GroupedPlan.allocate(work)
        plan.prepare(candidates)
        torch.npu.synchronize()
        check_plan(work, candidates, plan.plan)
        baseline = work.select(1, 2, candidates)[0]
        direct = work.select(1, 4, candidates)[0]
        grouped = plan.select(1)[0]
        torch.npu.synchronize()
        assert_equivalent(work, baseline, direct, 1, candidates)
        assert_equivalent(work, baseline, grouped, 1, candidates)
        if work.query.shape[1] > sum(counts):
            assert (grouped[sum(counts) :] == -1).all()
        print(
            json.dumps({"event": "correctness", "status": "pass", **case, **stats}),
            flush=True,
        )
        timing, outputs, graphs = time_runners(
            {
                "fast_mask_top512": partial(work.select, 1, 2, candidates),
                "single_query_direct": partial(work.select, 1, 4, candidates),
                "grouped_total": partial(plan.run, candidates),
                "grouped_prepared": partial(plan.select, 1),
                "plan_only": partial(plan.prepare, candidates),
            },
            args,
        )
        if args.mode == "graph":
            assert_equivalent(work, baseline, outputs["grouped_total"][0], 1, candidates)
            # Replay at fixed addresses with different candidates. This must
            # regenerate the plan inside the captured graph, not reuse a stale union.
            work.lengths.sub_(1)
            if len(counts) > 1:
                work.counts = [max(1, n - 1) for n in counts]
                work.starts.copy_(
                    torch.tensor(
                        [0] + list(torch.tensor(work.counts).cumsum(0).tolist()),
                        dtype=torch.int32,
                        device=work.starts.device,
                    )
                )
            work.refresh_metadata()
            changed, _ = make_candidates(work, 1.0 - overlap, holes)
            candidates.copy_(changed)
            graphs[2].replay()
            torch.npu.synchronize()
            expected = work.select(1, 2, candidates)[0]
            assert_equivalent(work, expected, outputs["grouped_total"][0], 1, candidates)
            check_plan(work, candidates, plan.plan)
            assert (outputs["grouped_total"][0][sum(work.counts) :] == -1).all()
            case["replay_counts"] = work.counts
        print(
            json.dumps({"scope": "one_consumer_including_plan", **case, **stats, **timing}),
            flush=True,
        )
        if args.mode == "eager" and counts == [512]:
            timing4, out4, graphs4 = time_runners(
                {
                    "fast_mask_top512": partial(four_baseline, work, candidates),
                    "grouped_total": partial(four_grouped, plan, candidates),
                },
                args,
            )
            for layer in range(1, 5):
                assert_equivalent(
                    work,
                    out4["fast_mask_top512"][layer - 1][0],
                    out4["grouped_total"][layer - 1][0],
                    layer,
                    candidates,
                )
            print(
                json.dumps(
                    {
                        "scope": "four_consumers_including_one_plan",
                        **case,
                        **stats,
                        **timing4,
                    }
                ),
                flush=True,
            )
            del out4, graphs4
        del work, candidates, plan, baseline, direct, grouped, outputs, graphs
        gc.collect()


if __name__ == "__main__":
    main()
