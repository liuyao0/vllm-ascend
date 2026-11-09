"""Exact packing, membership Top512, changing graph inputs and query counts."""

import gc
import json

import torch
from bench_common import assert_equivalent, make_workload
from mask_plan import MaskPlan as PackedPlan
from test_membership import candidates_for


def main():
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    for length, counts in [
        (31, [1]),
        (511, [7]),
        (32768, [4]),
        (50000, [6, 7, 8, 3, 1, 2]),
        (131072, [1, 6, 7, 8, 2, 9]),
        (131072, [512]),
    ]:
        torch.manual_seed(29)
        work = make_workload(length, counts, padding=3)
        work.weights[1::2].neg_()
        packed = PackedPlan.allocate(work)
        candidates = candidates_for(work, 1)
        candidates[sum(work.counts) :].fill_(2147483647)
        packed.prepare(candidates)
        actual = packed.select(1)[0]
        assert_equivalent(work, work.select(1, 2, candidates)[0], actual, 1, candidates)
        for _ in range(2):
            packed.prepare(candidates)
            packed.select(1)
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
            candidates[sum(work.counts) :].fill_(2147483647)
            work.key.neg_()
            work.kscale[:, ::17].mul_(1.125)
            graph.replay()
            torch.npu.synchronize()
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
