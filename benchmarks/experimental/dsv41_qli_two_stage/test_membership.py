"""Exercise all non-contiguous Q subsets, not just singleton/all-shared buckets."""

import json

import torch
from bench_common import assert_equivalent, make_workload
from benchmark_grouped import check_plan
from grouped_plan import GroupedPlan


def candidates_for(work, seed):
    generator = torch.Generator().manual_seed(seed)
    out = torch.full((work.query.shape[1], 1, 2048), -1, dtype=torch.int32)
    starts, lengths = work.starts.cpu().tolist(), work.lengths.cpu().tolist()
    for request, length in enumerate(lengths):
        for row in range(starts[request], starts[request + 1], 4):
            nq = min(4, starts[request + 1] - row)
            visible = length - starts[request + 1] + row + 1
            blocks = torch.randperm((visible + 7) // 8, generator=generator)
            sets = [[] for _ in range(nq)]
            offset = 0
            for mask in range(1, 1 << nq):
                size = 17 + (mask * 43 + seed) % 111
                subset = blocks[offset : offset + size]
                offset += size
                for q in range(nq):
                    if mask & (1 << q):
                        sets[q].append(subset)
            for q, subsets in enumerate(sets):
                ids = torch.cat(subsets)
                slots = torch.randperm(2048, generator=generator)[: ids.numel()]
                out[row + q, 0, slots] = ids.int()
    return out.to(work.query.device)


def main():
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    for length, counts in [(32768, [4]), (131072, [64]), (50000, [6, 7, 8, 3, 1, 2]), (131072, [512])]:
        work = make_workload(length, counts, padding=3)
        plan = GroupedPlan.allocate(work)
        candidates = candidates_for(work, 1)
        plan.prepare(candidates)
        actual = plan.select(1)[0]
        torch.npu.synchronize()
        check_plan(work, candidates, plan.plan)
        assert_equivalent(work, work.select(1, 2, candidates)[0], actual, 1, candidates)
        for _ in range(3):
            plan.run(candidates)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            plan.prepare(candidates)
            output = plan.select(1)[0]
        for seed in [7, 17]:
            candidates.copy_(candidates_for(work, seed))
            graph.replay()
            torch.npu.synchronize()
            check_plan(work, candidates, plan.plan)
            assert_equivalent(work, work.select(1, 2, candidates)[0], output, 1, candidates)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "length": length,
                    "counts": counts,
                    "checks": 3,
                    "all_subsets": True,
                    "graph_updates": 2,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
