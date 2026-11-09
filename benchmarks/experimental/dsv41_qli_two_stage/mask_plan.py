"""Cache the candidate block mask once for all four QLI consumers."""

from dataclasses import dataclass

import torch
from mask_plan_complete import build_mask as build_complete
from vllm.triton_utils import tl
from vllm.triton_utils import triton as tr


@tr.jit
def build_mask(candidates, mask, indices, starts, REQUESTS: tl.constexpr, STRIDE: tl.constexpr, P2: tl.constexpr):
    row = tl.program_id(0)
    lanes = tl.arange(0, 2048)
    live = row < tl.load(starts + REQUESTS)
    ids = tl.load(candidates + row * 2048 + lanes, live, -1)
    slots = tl.where(ids >= 0, ids, STRIDE - 1)
    tl.store(indices + row * 2048 + lanes, slots.to(tl.int64))
    for first in range(0, STRIDE, 2048):
        blocks = first + tl.arange(0, 2048)
        tl.store(mask + row * STRIDE + blocks, -1.0e30, blocks < STRIDE)


@dataclass
class MaskPlan:
    work: object
    mask: torch.Tensor
    indices: torch.Tensor

    @classmethod
    def allocate(cls, work):
        stride = tr.cdiv(work.table.shape[1] * 128, 2048) * 256 + 256
        return cls(
            work,
            torch.empty((work.query.shape[1], stride), dtype=torch.float32, device=work.query.device),
            torch.empty((work.query.shape[1], 2048), dtype=torch.int64, device=work.query.device),
        )

    @property
    def workspace_bytes(self):
        return self.mask.numel() * 4 + self.indices.numel() * 8

    def prepare(self, candidates):
        if self.work.max_length <= 2048 * 8:
            build_complete[(self.mask.shape[0],)](
                candidates,
                self.mask,
                self.work.starts,
                self.work.lengths,
                len(self.work.counts),
                tr.next_power_of_2(len(self.work.counts)),
                self.mask.shape[1],
                tr.next_power_of_2(self.mask.shape[1] - 256),
            )
            return
        build_mask[(self.mask.shape[0],)](
            candidates,
            self.mask,
            self.indices,
            self.work.starts,
            len(self.work.counts),
            self.mask.shape[1],
            tr.next_power_of_2(self.mask.shape[1]),
        )
        self.mask.scatter_(1, self.indices, float("inf"))

    def select(self, layer):
        return self.work.select(layer, 8, self.mask.view(torch.int32))
