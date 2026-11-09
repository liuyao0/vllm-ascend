"""Materialize membership buckets once; mode 7 consumes contiguous INT8 K."""

from dataclasses import dataclass

import torch
from grouped_plan import CAPACITY, CHUNK, INPUT_CAPACITY, STRIDE, GroupedPlan, _pack
from planner_stages import count_pack, layout
from vllm.triton_utils import tl
from vllm.triton_utils import triton as tr

TOKENS = CAPACITY * 8


@tr.jit
def gather_plan(
    key,
    scale,
    plan,
    packed_key,
    packed_scale,
    KS: tl.constexpr,
    SS: tl.constexpr,
    PAGE: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
):
    group, tile = tl.program_id(0), tl.program_id(1)
    first = tile * R
    total = tl.load(plan + group * S + 4)
    # Scalar block addresses let the compiler emit contiguous 1 KiB DMA,
    # instead of lowering a matrix of irregular addresses element by element.
    for block in range(first, tl.minimum(first + R, total)):
        physical = tl.load(plan + group * S + 64 + 2 * C + block)
        page, within = tl.maximum(physical, 0) // PAGE, tl.maximum(physical, 0) % PAGE
        lanes = tl.arange(0, 1024)
        address = page.to(tl.int64) * KS + within * 128
        values = tl.load(key + address + lanes, physical >= 0, 0)
        destination = (group.to(tl.int64) * C + block) * 8
        tl.store(packed_key + destination * 128 + lanes, values)
        tokens = tl.arange(0, 8)
        scales = tl.load(scale + page.to(tl.int64) * SS + within + tokens, physical >= 0, 0)
        tl.store(packed_scale + destination + tokens, scales)


@tr.jit
def copy_bucket_gather(
    packed,
    packed_physical,
    counts,
    plan,
    key,
    scale,
    packed_key,
    packed_scale,
    KS: tl.constexpr,
    SS: tl.constexpr,
    PAGE: tl.constexpr,
    NC: tl.constexpr,
    NCP2: tl.constexpr,
    P2: tl.constexpr,
    S: tl.constexpr,
    OC: tl.constexpr,
):
    group, chunk, bucket = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    size = tl.load(counts + (group * NC + chunk) * 16 + bucket)
    if size > 0:
        buckets = tl.arange(0, 16)
        local_sizes = tl.load(counts + (group * NC + chunk) * 16 + buckets)
        first = tl.sum(tl.where(buckets < bucket, local_sizes, 0))
        chunks = tl.arange(0, NCP2)
        sizes = tl.load(counts + (group * NC + chunks) * 16 + bucket, chunks < chunk, 0)
        base = tl.load(plan + group * S + 8 + 2 * bucket) + tl.sum(sizes)
        lanes = tl.arange(0, P2)
        selected = tl.load(packed + (group * NC + chunk) * P2 + first + lanes, lanes < size, 0).to(tl.float32)
        block = tl.where(
            bucket == 0, tl.floor(selected * 0.0625), selected - tl.floor(selected * (1.0 / 131072)) * 131072
        )
        cap = tl.where(bucket == 0, (selected - tl.floor(selected * 0.0625) * 16).to(tl.int32), 2139095040)
        physical = tl.load(packed_physical + (group * NC + chunk) * P2 + first + lanes, lanes < size, -1)
        tl.store(plan + group * S + 64 + base + lanes, block * 8, lanes < size)
        tl.store(plan + group * S + 64 + OC + base + lanes, cap, lanes < size)
        tl.store(plan + group * S + 64 + 2 * OC + base + lanes, physical, lanes < size)

        for local in range(size):
            phys = tl.load(packed_physical + (group * NC + chunk) * P2 + first + local)
            page, within = phys // PAGE, phys % PAGE
            values = tl.load(key + page.to(tl.int64) * KS + within * 128 + tl.arange(0, 1024))
            destination = (group.to(tl.int64) * OC + base + local) * 8
            tl.store(packed_key + destination * 128 + tl.arange(0, 1024), values)
            scales = tl.load(scale + page.to(tl.int64) * SS + within + tl.arange(0, 8))
            tl.store(packed_scale + destination + tl.arange(0, 8), scales)


@dataclass
class PackedPlan:
    work: object
    grouped: GroupedPlan
    key: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def allocate(cls, work):
        grouped = GroupedPlan.allocate(work)
        pages = grouped.plan.shape[0] * TOKENS // 128
        return cls(
            work,
            grouped,
            torch.empty((pages, 128, 1, 128), dtype=torch.int8, device=work.query.device),
            torch.zeros((pages, 128, 1), dtype=torch.float16, device=work.query.device),
        )

    @property
    def workspace_bytes(self):
        return self.key.numel() + self.scale.numel() * self.scale.element_size()

    def pack(self):
        gather_plan[(self.grouped.plan.shape[0], tr.cdiv(CAPACITY, 32))](
            self.work.key,
            self.work.kscale,
            self.grouped.plan,
            self.key,
            self.scale,
            self.work.key.stride(0),
            self.work.kscale.stride(0),
            self.work.key.shape[1],
            STRIDE,
            CAPACITY,
            32,
            multibuffer=True,
        )

    def prepare(self, candidates):
        g = self.grouped
        groups = g.plan.shape[0]
        chunks = tr.cdiv(CAPACITY, CHUNK)
        _pack[(groups, chunks)](
            candidates,
            self.work.starts,
            self.work.lengths,
            g.encoded,
            g.plan,
            len(self.work.counts),
            tr.next_power_of_2(len(self.work.counts)),
            INPUT_CAPACITY,
            STRIDE,
            CAPACITY,
            CHUNK,
        )
        torch.sort(g.encoded, dim=-1, out=(g.sorted_ids, g.sort_indices))
        count_pack[(groups, chunks)](
            g.sorted_ids,
            g.counts,
            g.packed,
            g.packed_physical,
            self.work.table,
            g.plan,
            INPUT_CAPACITY,
            chunks,
            CHUNK,
            STRIDE,
            self.work.table.shape[1],
            tr.next_power_of_2(self.work.table.shape[1]),
        )
        layout[(groups,)](g.counts, g.plan, chunks, tr.next_power_of_2(chunks), STRIDE)
        copy_bucket_gather[(groups, chunks, 16)](
            g.packed,
            g.packed_physical,
            g.counts,
            g.plan,
            self.work.key,
            self.work.kscale,
            self.key,
            self.scale,
            self.work.key.stride(0),
            self.work.kscale.stride(0),
            self.work.key.shape[1],
            chunks,
            tr.next_power_of_2(chunks),
            CHUNK,
            STRIDE,
            CAPACITY,
            multibuffer=True,
        )

    def select(self, layer):
        w = self.work
        return torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
            w.query[layer],
            self.key,
            w.weights[layer],
            w.qscale[layer],
            self.scale,
            512,
            2,
            block_table=w.table,
            metadata=w.metadata,
            candidate_topk_index=self.grouped.plan,
            candidate_mode=7,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
            **w.common(),
        )

    def run(self, candidates, layer=1):
        self.prepare(candidates)
        return self.select(layer)
