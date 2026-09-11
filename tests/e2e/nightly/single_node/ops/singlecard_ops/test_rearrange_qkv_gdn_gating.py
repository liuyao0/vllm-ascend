# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vllm-ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend._310p.ops.fla.fused_gdn_gating import fused_gdn_gating_pytorch
from vllm_ascend.ops import rearrange_qkv as rearrange_qkv_module
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.ops.triton.rearrange_qkv_gdn_gating import HAS_CORE_SCOPE
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

GDN_CONFIGS = [
    (1024, 1024, 3072),
    (128, 128, 256),
    (64, 64, 128),
]


@pytest.fixture(scope="module", autouse=True)
def device_setup():
    if get_ascend_device_type() not in {AscendDeviceType.A2, AscendDeviceType.A3}:
        pytest.skip("the fused rearrange/gating kernel is only built for A2 and A3")
    init_device_properties_triton()
    enable_custom_op()


def make_layer(q_dim, v_dim, tp_size=1):
    return SimpleNamespace(
        key_dim=q_dim * tp_size,
        value_dim=v_dim * tp_size,
        tp_size=tp_size,
        head_k_dim=128,
        head_v_dim=128,
    )


def reference_packed_qkv(mixed_qkv, dims):
    return torch.cat([part.reshape(-1) for part in mixed_qkv.split(dims, dim=-1)])


@pytest.mark.parametrize("impl", [rearrange_qkv_module.IMPL_ASCENDC, rearrange_qkv_module.IMPL_TRITON])
@pytest.mark.parametrize(("q_dim", "k_dim", "v_dim"), GDN_CONFIGS)
@pytest.mark.parametrize("tokens", [1, 3, 17, 129, 1024])
@pytest.mark.parametrize("head_dtype", [torch.float32, torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_rearrange_matches_reference(impl, q_dim, k_dim, v_dim, tokens, head_dtype):
    if impl == rearrange_qkv_module.IMPL_TRITON and not HAS_CORE_SCOPE:
        pytest.skip("installed triton-ascend has no cube/vector scope support")
    device = "npu"
    num_heads = v_dim // 128
    mixed_qkv = torch.randn(tokens, q_dim + k_dim + v_dim, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(num_heads, dtype=head_dtype, device=device)
    dt_bias = torch.randn(num_heads, dtype=head_dtype, device=device)
    a = torch.randn(tokens, num_heads, dtype=torch.bfloat16, device=device)
    b = torch.randn(tokens, num_heads, dtype=torch.bfloat16, device=device)
    layer = make_layer(q_dim, v_dim)

    with patch.object(rearrange_qkv_module, "_FUSED_IMPL_CACHE", impl):
        query, key, value, g, beta = rearrange_qkv_module.rearrange_mixed_qkv_and_fused_gdn_gating(
            layer, mixed_qkv, A_log, a, b, dt_bias
        )

    expected = reference_packed_qkv(mixed_qkv, (q_dim, k_dim, v_dim))
    actual = torch.cat([query.reshape(-1), key.reshape(-1), value.reshape(-1)])
    assert torch.equal(actual.view(torch.int16).cpu(), expected.view(torch.int16).cpu())
    assert g.shape == (1, tokens, num_heads)
    assert beta.shape == (1, tokens, num_heads)


@pytest.mark.parametrize("impl", [rearrange_qkv_module.IMPL_ASCENDC, rearrange_qkv_module.IMPL_TRITON])
@pytest.mark.parametrize(("q_dim", "k_dim", "v_dim"), GDN_CONFIGS)
@pytest.mark.parametrize("tokens", [1, 17, 129, 1024])
@pytest.mark.parametrize("head_dtype", [torch.float32, torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_gating_matches_torch_reference(impl, q_dim, k_dim, v_dim, tokens, head_dtype):
    if impl == rearrange_qkv_module.IMPL_TRITON and not HAS_CORE_SCOPE:
        pytest.skip("installed triton-ascend has no cube/vector scope support")
    device = "npu"
    num_heads = v_dim // 128
    mixed_qkv = torch.randn(tokens, q_dim + k_dim + v_dim, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(num_heads, dtype=head_dtype, device=device)
    dt_bias = torch.randn(num_heads, dtype=head_dtype, device=device)
    a = torch.randn(tokens, num_heads, dtype=torch.bfloat16, device=device)
    b = torch.randn(tokens, num_heads, dtype=torch.bfloat16, device=device)
    layer = make_layer(q_dim, v_dim)

    with patch.object(rearrange_qkv_module, "_FUSED_IMPL_CACHE", impl):
        _, _, _, g, beta = rearrange_qkv_module.rearrange_mixed_qkv_and_fused_gdn_gating(
            layer, mixed_qkv, A_log, a, b, dt_bias
        )

    ref_g, ref_beta = fused_gdn_gating_pytorch(A_log=A_log, a=a, b=b, dt_bias=dt_bias)
    torch.testing.assert_close(g.to(torch.float32).cpu(), ref_g.to(torch.float32).cpu(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        beta.to(torch.float32).cpu(), ref_beta.to(torch.float32).cpu(), rtol=1e-2, atol=1e-2
    )


@pytest.mark.parametrize("impl", [rearrange_qkv_module.IMPL_ASCENDC, rearrange_qkv_module.IMPL_TRITON])
@pytest.mark.parametrize("tokens", [17, 321, 4096])
@torch.inference_mode()
def test_fused_matches_separate_ops(impl, tokens):
    """The fused kernel must reproduce the two dedicated ops bit for bit on qkv."""
    if impl == rearrange_qkv_module.IMPL_TRITON and not HAS_CORE_SCOPE:
        pytest.skip("installed triton-ascend has no cube/vector scope support")
    device = "npu"
    q_dim = k_dim = 1024
    v_dim = 3072
    num_heads = v_dim // 128
    mixed_qkv = torch.randn(tokens, q_dim + k_dim + v_dim, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(num_heads, dtype=torch.bfloat16, device=device)
    dt_bias = torch.randn(num_heads, dtype=torch.bfloat16, device=device)
    a = torch.randn(tokens, num_heads, dtype=torch.bfloat16, device=device)
    b = torch.randn(tokens, num_heads, dtype=torch.bfloat16, device=device)
    layer = make_layer(q_dim, v_dim)

    expected_qkv = torch.ops._C_ascend.npu_rearrange_qkv(mixed_qkv, q_dim, k_dim, v_dim)
    expected_g, expected_beta = fused_gdn_gating_patch(A_log=A_log, a=a, b=b, dt_bias=dt_bias)

    with patch.object(rearrange_qkv_module, "_FUSED_IMPL_CACHE", impl):
        query, key, value, g, beta = rearrange_qkv_module.rearrange_mixed_qkv_and_fused_gdn_gating(
            layer, mixed_qkv, A_log, a, b, dt_bias
        )

    actual_qkv = torch.cat([query.reshape(-1), key.reshape(-1), value.reshape(-1)])
    assert torch.equal(actual_qkv.view(torch.int16).cpu(), expected_qkv.view(torch.int16).cpu())
    torch.testing.assert_close(g.cpu(), expected_g.cpu(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        beta.to(torch.float32).cpu(), expected_beta.to(torch.float32).cpu(), rtol=1e-2, atol=1e-2
    )


@pytest.mark.parametrize(
    ("q_dim", "v_dim", "tp_size"),
    [(128, 256, 1), (512, 1536, 4), (1024, 3072, 2), (2048, 6144, 1)],
)
@torch.inference_mode()
def test_gdn_dispatch_uses_fused_kernel(q_dim, v_dim, tp_size):
    device = "npu"
    tokens = 37
    num_heads = v_dim // 128
    mixed_qkv = torch.randn(tokens, 2 * q_dim + v_dim, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(num_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_heads, dtype=torch.float32, device=device)
    a = torch.randn(tokens, num_heads, dtype=torch.bfloat16, device=device)
    b = torch.randn(tokens, num_heads, dtype=torch.bfloat16, device=device)

    def unexpected_fallback(_):
        pytest.fail("the supported GDN layout did not use the fused kernel")

    layer = SimpleNamespace(
        key_dim=q_dim * tp_size,
        value_dim=v_dim * tp_size,
        tp_size=tp_size,
        head_k_dim=128,
        head_v_dim=128,
        rearrange_mixed_qkv=unexpected_fallback,
    )
    with patch.object(rearrange_qkv_module, "_FUSED_IMPL_CACHE", rearrange_qkv_module.IMPL_ASCENDC):
        query, key, value, g, beta = rearrange_qkv_module.rearrange_mixed_qkv_and_fused_gdn_gating(
            layer, mixed_qkv, A_log, a, b, dt_bias
        )
    assert query.shape == (1, tokens, q_dim // 128, 128)
    assert key.shape == (1, tokens, q_dim // 128, 128)
    assert value.shape == (1, tokens, v_dim // 128, 128)
    assert g.shape == (1, tokens, num_heads)
    assert beta.shape == (1, tokens, num_heads)
