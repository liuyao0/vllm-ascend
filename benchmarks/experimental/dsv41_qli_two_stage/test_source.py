import gc
import json

import torch
from bench_common import assert_equivalent, make_workload


def compare(w, ref, out):
    live = sum(w.counts)
    torch.testing.assert_close(ref[2][:live], out[2][:live], rtol=0, atol=0)
    assert_equivalent(w, ref[0], out[0], 0, None)


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
    torch.manual_seed(91)
    w = make_workload(length, counts, padding=3)
    compare(w, w.select(0, 1), w.select(0, 9))
    for _ in range(3):
        w.select(0, 9)
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        out = w.select(0, 9)
    for update in range(2):
        if update and len(counts) > 1:
            w.counts = [max(1, n - 1) for n in counts]
            w.starts.copy_(
                torch.tensor([0] + torch.tensor(w.counts).cumsum(0).tolist(), device=w.query.device, dtype=torch.int32)
            )
        w.lengths.sub_(1)
        w.refresh_metadata()
        w.key.neg_()
        w.weights[0].neg_()
        w.kscale[:, ::17].mul_(1.125)
        g.replay()
        torch.npu.synchronize()
        compare(w, w.select(0, 1), out)
    print(
        json.dumps(
            {
                "status": "pass",
                "length": length,
                "counts": counts,
                "replay_counts": w.counts,
                "candidate_blocks_exact": True,
                "source_token_top512": True,
                "graph_updates": 2,
            }
        ),
        flush=True,
    )
    del w, g, out
    gc.collect()
