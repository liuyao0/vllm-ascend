# SPDX-License-Identifier: Apache-2.0
"""Compare baseline and extracted fast-mask/Top512 builds through native mode 2.

Run --phase baseline before rebuilding, then --phase optimized with the same
--reference directory. Each query has independently sampled candidate blocks.
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env

# The native extension loads its vendor libraries at import time.
bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # noqa: E402, F401

from vllm_ascend.ops.triton.prepare_indexer_indices import (  # noqa: E402
    prepare_indexer_indices,
)
from vllm_ascend.ops.triton.triton_utils import (  # noqa: E402
    init_device_properties_triton,
)


@dataclass
class Workload:
    query: torch.Tensor
    weights: torch.Tensor
    qscale: torch.Tensor
    key: torch.Tensor
    kscale: torch.Tensor
    table: torch.Tensor
    starts: torch.Tensor
    lengths: torch.Tensor
    positions: torch.Tensor
    metadata: torch.Tensor
    max_query: int
    max_length: int
    counts: list[int]

    def common(self):
        return {
            "cu_seqlens_q": self.starts,
            "seqused_k": self.lengths,
            "max_seqlen_q": self.max_query,
            "layout_q": "TND",
            "layout_k": "PA_BBND",
            "mask_mode": 3,
            "cmp_ratio": 1,
        }

    def refresh_metadata(self):
        # Benchmark setup only: match the model's per-query absolute positions.
        starts, lengths = self.starts.cpu().tolist(), self.lengths.cpu().tolist()
        positions = torch.full((self.query.shape[1],), -1, dtype=torch.int64)
        for request, length in enumerate(lengths):
            start, end = starts[request : request + 2]
            positions[start:end] = torch.arange(length - (end - start), length)
        self.positions.copy_(positions)
        metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            32,
            1,
            128,
            512,
            2,
            batch_size=len(self.counts),
            max_seqlen_k=self.max_length,
            device=str(self.query.device),
            **self.common(),
        )
        self.metadata.copy_(metadata)

    def select(self, layer, mode, candidates=None):
        return torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
            self.query[layer],
            self.key,
            self.weights[layer],
            self.qscale[layer],
            self.kscale,
            512,
            2,
            block_table=self.table,
            metadata=self.metadata,
            candidate_topk_index=candidates,
            candidate_mode=mode,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
            **self.common(),
        )

    def run(self, mode):
        source, _, candidates = self.select(0, 1)
        outputs = [source]
        for layer in range(1, 5):
            outputs.append(self.select(layer, mode, candidates)[0])
        # Match the current model: visibility filtering, chronological order
        # and valid lengths are fused in prepare_indexer_indices, not torch.sort.
        return tuple(
            prepare_indexer_indices(
                output.squeeze(1),
                self.positions,
                1,
                lengths_output=torch.empty((output.shape[0], 1), dtype=torch.int32, device=output.device),
            ).unsqueeze(1)
            for output in outputs
        )


def make_workload(length, counts, padding=0):
    init_device_properties_triton()
    requests, rows = len(counts), sum(counts) + padding
    pages = (length + 127) // 128
    storage = torch.empty(requests * pages * 147712, dtype=torch.int8, device="npu")
    key = storage.as_strided((requests * pages, 128, 1, 128), (147712, 128, 128, 1), 131072)
    scale = storage.view(torch.float16).as_strided((requests * pages, 128, 1), (73856, 1, 1), 73728)
    key.copy_(torch.randint(-90, 91, key.shape, dtype=torch.int8, device="npu"))
    scale.copy_(torch.rand(scale.shape, dtype=torch.float16, device="npu") * 0.01 + 0.005)
    starts = torch.tensor(
        [0] + list(torch.tensor(counts).cumsum(0).tolist()),
        dtype=torch.int32,
        device="npu",
    )
    work = Workload(
        query=torch.randint(-90, 91, (5, rows, 32, 128), dtype=torch.int8, device="npu"),
        weights=torch.rand(5, rows, 32, dtype=torch.float16, device="npu"),
        qscale=torch.full((5, rows, 32), 0.01, dtype=torch.float16, device="npu"),
        key=key,
        kscale=scale,
        table=torch.randperm(requests * pages, dtype=torch.int32, device="npu").reshape(requests, pages),
        starts=starts,
        lengths=torch.full((requests,), length, dtype=torch.int32, device="npu"),
        positions=torch.empty(rows, dtype=torch.int64, device="npu"),
        metadata=torch.empty(1024, dtype=torch.int32, device="npu"),
        max_query=max(counts),
        max_length=length,
        counts=counts,
    )
    work.refresh_metadata()
    return work


def assert_equivalent(work, reference_output, actual_output, layer, candidates):
    """Compare IDs when exact; on ties compare native-rounded selected scores.

    Copies only selected keys to CPU, not the full cache. Invalid rows and the
    -1e30 shortage-fill class are checked separately from numeric tolerances.
    """
    reference, result = (
        reference_output.cpu().reshape(-1, 512),
        actual_output.cpu().reshape(-1, 512),
    )
    candidate_rows = candidates.cpu().reshape(-1, 2048) if candidates is not None else None
    starts, lengths = work.starts.cpu().tolist(), work.lengths.cpu().tolist()
    for request in range(len(work.counts)):
        for row in range(starts[request], starts[request + 1]):
            visible = lengths[request] - starts[request + 1] + row + 1
            expected, actual = reference[row], result[row]
            valid = actual[actual >= 0]
            assert valid.unique().numel() == valid.numel(), (row, "duplicate positions")
            assert (valid < visible).all(), (row, "future position")
            assert (actual == -1).sum() == (expected == -1).sum(), (
                row,
                "padding count",
            )
            assert (actual >= -1).all()
            if torch.equal(expected.sort().values, actual.sort().values):
                continue
            union = torch.cat((expected[expected >= 0], valid))
            ids = union.to(work.table.device).long()
            pages = work.table[request, ids // 128].long()
            keys = work.key[pages, ids % 128, 0].cpu().float()
            scales = work.kscale[pages, ids % 128, 0].cpu().float()
            q = work.query[layer, row].cpu().float()
            activation = ((q @ keys.T) / 1024).clamp_min(0).half().float()
            weights = (work.weights[layer, row].cpu() * work.qscale[layer, row].cpu()).float()
            values = (weights[:, None] * activation).sum(0) * scales
            if candidate_rows is not None:
                candidate = torch.isin(union // 8, candidate_rows[row])
                values[~candidate] = -1e30
            split = int((expected >= 0).sum())
            torch.testing.assert_close(
                values[:split].sort().values,
                values[split:].sort().values,
                rtol=3e-5,
                atol=1e-5,
                msg=f"layer={layer}, row={row}",
            )


def measure(run, repeats):
    start, end = (
        torch.npu.Event(enable_timing=True),
        torch.npu.Event(enable_timing=True),
    )
    torch.npu.synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(repeats):
        run()
    end.record()
    end.synchronize()
    return {
        "device_us": start.elapsed_time(end) * 1000 / repeats,
        "wall_us": (time.perf_counter() - wall_start) * 1e6 / repeats,
    }


def make_candidates(work, pattern):
    candidates = torch.full((work.query.shape[1], 1, 2048), -1, dtype=torch.int32)
    starts, lengths = work.starts.cpu().tolist(), work.lengths.cpu().tolist()
    for request, length in enumerate(lengths):
        for row in range(starts[request], starts[request + 1]):
            visible = length - starts[request + 1] + row + 1
            blocks = (visible + 7) // 8
            ids = torch.randperm(blocks)[:2048]
            if pattern == "clustered":
                # Sample four regions, with independent IDs per query row.
                region = max(1, blocks // 16)
                allowed = torch.arange(blocks)
                centers = torch.randperm(blocks)[:4]
                distance = (allowed[:, None] - centers[None, :]).abs()
                allowed = allowed[(distance < region).any(dim=1)]
                ids = allowed[torch.randperm(allowed.numel())[:2048]]
            if pattern == "holes":
                ids = ids[:1024]
                candidates[row, 0, : 2 * ids.numel() : 2] = ids.int()
            else:
                candidates[row, 0, : ids.numel()] = ids.int()
    return candidates.to(work.query.device)


def timed(run, args):
    for _ in range(args.warmup):
        result = run()
    torch.npu.synchronize()
    if args.mode == "graph":
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = run()
        replay = graph.replay
    else:
        replay = run
    samples = [measure(replay, args.repeats) for _ in range(args.samples)]
    return {
        "samples": samples,
        "device_us": statistics.median(sample["device_us"] for sample in samples),
        "wall_us": statistics.median(sample["wall_us"] for sample in samples),
    }, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["baseline", "optimized"])
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 32768, 65536, 131072])
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=[8],
        help="Verification queries in each request",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["uniform", "holes", "clustered"],
        default=["uniform", "holes", "clustered"],
    )
    parser.add_argument("--mode", choices=["eager", "graph"], default="graph")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    args.reference.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "device": torch.npu.get_device_name(),
                "torch": torch.__version__,
                "phase": args.phase,
            }
        ),
        flush=True,
    )
    for length in args.lengths:
        # Both builds reconstruct identical Q/K, physical pages and candidates.
        torch.manual_seed(71)
        work = make_workload(length, args.counts)
        _, _, source_candidates = work.select(0, 1)
        for pattern in args.patterns:
            torch.manual_seed(72)
            candidates = make_candidates(work, pattern)
            case = {
                "length": length,
                "counts": args.counts,
                "pattern": pattern,
                "mode": args.mode,
            }
            counts_name = "-".join(map(str, args.counts))
            reference_path = args.reference / f"{length}_{counts_name}_{pattern}_{args.mode}.pt"
            timing, result = timed(partial(work.select, 1, 2, candidates), args)
            timings = {"consumer": timing}
            snapshot = {
                "case": case,
                "consumer": result[0].cpu(),
                "candidates": candidates.cpu(),
            }
            if pattern == args.patterns[0]:
                timings["group"], group = timed(partial(work.run, 2), args)
                snapshot["group"] = [output.cpu() for output in group]
                snapshot["source_candidates"] = source_candidates.cpu()
            if args.phase == "baseline":
                snapshot["timings"] = timings
                torch.save(snapshot, reference_path)
                speedup = None
            else:
                reference = torch.load(reference_path, map_location="cpu", weights_only=True)
                assert reference["case"] == case
                assert torch.equal(reference["candidates"], snapshot["candidates"])
                assert_equivalent(work, reference["consumer"], snapshot["consumer"], 1, candidates)
                if "group" in snapshot:
                    assert torch.equal(reference["source_candidates"], snapshot["source_candidates"])
                    assert torch.equal(reference["group"][0], snapshot["group"][0])
                    for layer in range(1, 5):
                        assert_equivalent(
                            work,
                            reference["group"][layer],
                            snapshot["group"][layer],
                            layer,
                            source_candidates,
                        )
                speedup = {
                    scope: reference["timings"][scope]["device_us"] / value["device_us"]
                    for scope, value in timings.items()
                }
            print(
                json.dumps(
                    {
                        **case,
                        "requests": len(args.counts),
                        "rows": sum(args.counts),
                        "phase": args.phase,
                        "timings": timings,
                        "speedup": speedup,
                    }
                ),
                flush=True,
            )
        del work, source_candidates, candidates, result, snapshot


if __name__ == "__main__":
    main()
