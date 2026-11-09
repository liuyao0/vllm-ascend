"""Cache the candidate block mask once for all four QLI consumers."""

from dataclasses import dataclass

import torch
from vllm.triton_utils import tl
from vllm.triton_utils import triton as tr


@tr.jit
def build_mask(
    candidates, mask, starts, lengths, REQUESTS: tl.constexpr, RP2: tl.constexpr, STRIDE: tl.constexpr, P2: tl.constexpr
):
    row = tl.program_id(0)
    lanes = tl.arange(0, 2048)
    live = row < tl.load(starts + REQUESTS)
    ids = tl.load(candidates + row * 2048 + lanes, live, -1)
    requests = tl.arange(0, RP2)
    ends = tl.load(starts + requests + 1, requests < REQUESTS, 2147483647)
    request = tl.minimum(tl.sum((row >= ends).to(tl.int32)), REQUESTS - 1)
    visible = tl.where(live, tl.load(lengths + request) - tl.load(starts + request + 1) + row + 1, 0)
    visible_blocks = tl.cdiv(visible, 8)
    valid = (ids >= 0) & (ids < STRIDE - 256)
    complete = (tl.sum(valid.to(tl.int32)) == visible_blocks) & (tl.max(ids) < visible_blocks)
    blocks = tl.arange(0, P2)
    if complete:
        values = tl.where(blocks < visible_blocks, float("inf"), -1.0e30)
    else:
        counts = tl.histogram(ids, P2, mask=valid)
        values = tl.where(counts > 0, float("inf"), -1.0e30)
    tl.store(mask + row * STRIDE + blocks, values, blocks < STRIDE - 256)


@dataclass
class MaskPlan:
    work: object
    mask: torch.Tensor

    @classmethod
    def allocate(cls, work):
        stride = tr.cdiv(work.table.shape[1] * 128, 2048) * 256 + 256
        return cls(work, torch.empty((work.query.shape[1], stride), dtype=torch.float32, device=work.query.device))

    @property
    def workspace_bytes(self):
        return self.mask.numel() * 4

    def prepare(self, candidates):
        build_mask[(self.mask.shape[0],)](
            candidates,
            self.mask,
            self.work.starts,
            self.work.lengths,
            len(self.work.counts),
            tr.next_power_of_2(len(self.work.counts)),
            self.mask.shape[1],
            tr.next_power_of_2(self.mask.shape[1] - 256),
        )

    def select(self, layer):
        return self.work.select(layer, 8, self.mask.view(torch.int32))
