# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vllm-ascend project
"""Compare the fused GDN prologue (DMA QKV rearrange || gating) with the split ops."""

import argparse

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops import rearrange_qkv as rearrange_qkv_module
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op


def measure(fn, warmup, iterations, use_graph):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    if use_graph:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = fn()
        run = graph.replay
    else:
        run = fn

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        run()
    end.record()
    end.synchronize()
    if use_graph:
        del output
    return start.elapsed_time(end) * 1000 / iterations


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 16, 64, 256, 1024, 4096])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--q-dim", type=int, default=1024, help="Per-rank Q/K width")
    parser.add_argument("--v-dim", type=int, default=3072, help="Per-rank V width")
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    enable_custom_op()
    init_device_properties_triton()
    num_heads = args.v_dim // args.head_dim
    layer = type(
        "Layer",
        (),
        {
            "key_dim": args.q_dim,
            "value_dim": args.v_dim,
            "tp_size": 1,
            "head_k_dim": args.head_dim,
            "head_v_dim": args.head_dim,
            "rearrange_mixed_qkv": lambda self, x: None,
        },
    )()

    print("tokens,q_dim,v_dim,split_us,ascendc_fused_us,triton_fused_us")
    for tokens in args.tokens:
        mixed_qkv = torch.randn(tokens, 2 * args.q_dim + args.v_dim, device="npu", dtype=torch.bfloat16)
        a = torch.randn(tokens, num_heads, device="npu", dtype=torch.bfloat16)
        b = torch.randn(tokens, num_heads, device="npu", dtype=torch.bfloat16)
        A_log = torch.randn(num_heads, device="npu", dtype=torch.bfloat16)
        dt_bias = torch.randn(num_heads, device="npu", dtype=torch.bfloat16)

        def split():
            g, beta = fused_gdn_gating_patch(A_log=A_log, a=a, b=b, dt_bias=dt_bias)
            return torch.ops._C_ascend.npu_rearrange_qkv(mixed_qkv, args.q_dim, args.q_dim, args.v_dim), g, beta

        def fused(impl):
            with rearrange_qkv_module._impl_override(impl):
                return rearrange_qkv_module.rearrange_mixed_qkv_and_fused_gdn_gating(
                    layer, mixed_qkv, A_log, a, b, dt_bias
                )

        split_us = measure(split, args.warmup, args.iters, args.graph)
        ascendc_us = measure(lambda: fused(rearrange_qkv_module.IMPL_ASCENDC), args.warmup, args.iters, args.graph)
        triton_us = measure(lambda: fused(rearrange_qkv_module.IMPL_TRITON), args.warmup, args.iters, args.graph)
        print(f"{tokens},{args.q_dim},{args.v_dim},{split_us:.4f},{ascendc_us:.4f},{triton_us:.4f}")


if __name__ == "__main__":
    main()
