# SPDX-License-Identifier: Apache-2.0
"""Device-only four-query candidate union for the experimental A2 QLI mode 6.

Plan layout: 64 header int32s then 9216 origins, membership masks, physical
token starts. Header 0:6 holds row/count/request/visible/total/last bucket;
header 8+2*b holds bucket b start and padded count. Bucket zero is the prefix. No K contents are copied to GM workspace.
"""

from dataclasses import dataclass

import torch
from planner_stages import copy_bucket, count_pack, layout
from vllm.triton_utils import tl
from vllm.triton_utils import triton as tr

INPUT_CAPACITY = 9248  # eight leading sentinels + 9216 vector-readable entries + tail
CAPACITY = 4 * 2048 + 16 * 64
HEADER = 64
STRIDE = HEADER + 3 * CAPACITY
CHUNK = 1024


@tr.jit
def _pack(
    candidates,
    starts,
    lengths,
    encoded,
    plan,
    REQUESTS: tl.constexpr,
    RP2: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    OC: tl.constexpr,
    P2: tl.constexpr,
):
    group = tl.program_id(0)
    reqs = tl.arange(0, RP2)
    begins = tl.load(starts + reqs, reqs < REQUESTS, 2147483647)
    ends = tl.load(starts + reqs + 1, reqs < REQUESTS, 2147483647)
    per_request = tl.where(reqs < REQUESTS, tl.cdiv(ends - begins, 4), 0)
    bases = tl.cumsum(per_request) - per_request
    total_groups = tl.sum(per_request)
    request = tl.maximum(tl.sum(((reqs < REQUESTS) & (group >= bases)).to(tl.int32)) - 1, 0)
    begin = tl.load(starts + request)
    end = tl.load(starts + request + 1)
    request_base = tl.sum(tl.where(reqs == request, bases, 0))
    row = begin + (group - request_base) * 4
    nq = tl.minimum(tl.maximum(end - row, 0), 4)
    visible = tl.load(lengths + request) - (end - row) + 1
    chunk = tl.program_id(1)
    i = chunk * P2 + tl.arange(0, P2)
    # Each program reads exactly half of one query row, or the 64-block prefix.
    # Make the query predicate scalar so GM loads remain vector DMA on A2.
    if chunk < 8:
        q = chunk // 2
        candidate = tl.load(candidates + row * 2048 + i, q < nq, -1)
        block = candidate
        limit = visible + q
        bits = 1 << q
        valid = (nq > 0) & (block >= 0) & (block < tl.cdiv(limit, 8))
    else:
        block = i - 8192
        bits = 0
        valid = (nq > 0) & (block < 64) & (block < tl.cdiv(visible + nq - 1, 8))
    value = tl.where(valid, block * 16 + bits, 16777215)
    tl.store(encoded + group * C + 8 + i, value)
    tl.store(plan + group * S + 64 + i, -8)
    tl.store(plan + group * S + 64 + OC + i, -8388608)
    tl.store(plan + group * S + 64 + 2 * OC + i, -1)
    if chunk == 0:
        lanes = tl.arange(0, P2)
        tl.store(encoded + group * C + lanes, -1, lanes < 8)
        tl.store(encoded + group * C + 9224 + lanes, 16777215, lanes < 24)
    if chunk == 0:
        tl.store(plan + group * S, row)
        tl.store(plan + group * S + 1, nq)
        tl.store(plan + group * S + 2, request)
        tl.store(plan + group * S + 3, visible)
        tl.store(plan + group * S + 6, total_groups)


@dataclass
class GroupedPlan:
    work: object
    plan: torch.Tensor
    encoded: torch.Tensor
    sorted_ids: torch.Tensor
    sort_indices: torch.Tensor
    counts: torch.Tensor
    packed: torch.Tensor
    packed_physical: torch.Tensor

    @classmethod
    def allocate(cls, work):
        groups = tr.cdiv(work.query.shape[1], 4) + len(work.counts)
        options = {"device": work.query.device, "dtype": torch.int32}
        encoded = torch.empty((groups, INPUT_CAPACITY), device=work.query.device, dtype=torch.float32)
        return cls(
            work,
            torch.empty((groups, STRIDE), **options),
            encoded,
            torch.empty_like(encoded),
            torch.empty_like(encoded, dtype=torch.int64),
            torch.empty((groups, tr.cdiv(CAPACITY, CHUNK), 16), **options),
            torch.empty((groups, tr.cdiv(CAPACITY, CHUNK), CHUNK), **options),
            torch.empty((groups, tr.cdiv(CAPACITY, CHUNK), CHUNK), **options),
        )

    def prepare(self, candidates):
        groups = self.plan.shape[0]
        chunks = tr.cdiv(CAPACITY, CHUNK)
        _pack[(groups, tr.cdiv(CAPACITY, CHUNK))](
            candidates,
            self.work.starts,
            self.work.lengths,
            self.encoded,
            self.plan,
            len(self.work.counts),
            tr.next_power_of_2(len(self.work.counts)),
            INPUT_CAPACITY,
            STRIDE,
            CAPACITY,
            CHUNK,
        )
        torch.sort(self.encoded, dim=-1, out=(self.sorted_ids, self.sort_indices))
        count_pack[(groups, chunks)](
            self.sorted_ids,
            self.counts,
            self.packed,
            self.packed_physical,
            self.work.table,
            self.plan,
            INPUT_CAPACITY,
            chunks,
            CHUNK,
            STRIDE,
            self.work.table.shape[1],
            tr.next_power_of_2(self.work.table.shape[1]),
        )
        layout[(groups,)](self.counts, self.plan, chunks, tr.next_power_of_2(chunks), STRIDE)
        copy_bucket[(groups, chunks, 16)](
            self.packed,
            self.packed_physical,
            self.counts,
            self.plan,
            chunks,
            tr.next_power_of_2(chunks),
            CHUNK,
            STRIDE,
            CAPACITY,
        )
        return self.plan

    def select(self, layer):
        return self.work.select(layer, 6, self.plan)

    def run(self, candidates, layer=1):
        self.prepare(candidates)
        return self.select(layer)
