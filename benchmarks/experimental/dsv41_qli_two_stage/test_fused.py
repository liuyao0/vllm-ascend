"""Exact packing, membership Top512, changing graph inputs and query counts."""

import gc
import json

import torch
from bench_common import assert_equivalent, make_workload
from benchmark_grouped import check_plan
from grouped_plan import CAPACITY, HEADER
from packed_plan import TOKENS, PackedPlan
from test_membership import candidates_for


def check_packing(work, packed):
    plan = packed.grouped.plan.cpu()
    groups, slots, physical = [], [], []
    for group, record in enumerate(plan):
        if not record[1]:
            continue
        for bucket in range(16):
            start, size = record[8 + 2 * bucket : 10 + 2 * bucket].tolist()
            if not size:
                continue
            for slot in {start, start + size - 1}:
                groups.append(group)
                slots.append(slot)
                physical.append(int(record[HEADER + 2 * CAPACITY + slot]))
    g = torch.tensor(groups, device=work.query.device)
    s = torch.tensor(slots, device=work.query.device)
    phys = torch.tensor(physical, device=work.query.device)
    expected_k = work.key[phys.clamp_min(0) // 128, phys.clamp_min(0) % 128, 0]
    expected_s = work.kscale[phys.clamp_min(0) // 128, phys.clamp_min(0) % 128, 0]
    expected_k[phys < 0] = 0
    expected_s[phys < 0] = 0
    torch.testing.assert_close(
        packed.key.view(-1, TOKENS, 128)[g, s * 8][phys >= 0], expected_k[phys >= 0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        packed.scale.view(-1, TOKENS)[g, s * 8][phys >= 0], expected_s[phys >= 0], rtol=0, atol=0
    )


def main():
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    for length, counts in [
        (511, [7]),
        (32768, [4]),
        (50000, [6, 7, 8, 3, 1, 2]),
        (131072, [1, 6, 7, 8, 2, 9]),
        (131072, [512]),
    ]:
        torch.manual_seed(29)
        work = make_workload(length, counts, padding=3)
        packed = PackedPlan.allocate(work)
        candidates = candidates_for(work, 1)
        packed.prepare(candidates)
        check_plan(work, candidates, packed.grouped.plan)
        check_packing(work, packed)
        actual = packed.select(1)[0]
        assert_equivalent(work, work.select(1, 2, candidates)[0], actual, 1, candidates)
        for _ in range(2):
            packed.run(candidates)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            packed.prepare(candidates)
            outputs = tuple(packed.select(layer)[0] for layer in range(1, 5))
        for seed in [7, 17]:
            if seed == 17 and len(counts) > 1:
                work.counts = [max(1, n - 1) for n in counts]
                work.starts.copy_(
                    torch.tensor(
                        [0] + torch.tensor(work.counts).cumsum(0).tolist(), device=work.query.device, dtype=torch.int32
                    )
                )
            work.lengths.sub_(1)
            work.refresh_metadata()
            candidates.copy_(candidates_for(work, seed))
            work.key.neg_()
            work.kscale[:, ::17].mul_(1.125)
            graph.replay()
            torch.npu.synchronize()
            check_plan(work, candidates, packed.grouped.plan)
            check_packing(work, packed)
            for layer, actual in enumerate(outputs, 1):
                assert_equivalent(work, work.select(layer, 2, candidates)[0], actual, layer, candidates)
                assert bool((actual[sum(work.counts) :] == -1).all())
        print(
            json.dumps(
                {
                    "status": "pass",
                    "length": length,
                    "counts": counts,
                    "replay_counts": work.counts,
                    "all_memberships": True,
                    "graph_updates": 2,
                    "workspace_bytes": packed.workspace_bytes,
                }
            ),
            flush=True,
        )
        del work, packed, candidates, outputs, actual, graph
        gc.collect()


if __name__ == "__main__":
    main()
