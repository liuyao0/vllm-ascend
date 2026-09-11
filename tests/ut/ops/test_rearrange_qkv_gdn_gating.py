# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vllm-ascend project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ops import rearrange_qkv as rearrange_qkv_module
from vllm_ascend._310p.ops.fla.fused_gdn_gating import fused_gdn_gating_pytorch


def make_layer(**kwargs):
    attributes = {
        "key_dim": 2048,
        "value_dim": 6144,
        "tp_size": 2,
        "head_k_dim": 128,
        "head_v_dim": 128,
    }
    attributes.update(kwargs)
    return SimpleNamespace(
        **attributes,
        rearrange_mixed_qkv=Mock(return_value=(None, None, None)),
    )


def make_tensor(shape, dtype=torch.bfloat16, contiguous=True, device="npu:0"):
    return SimpleNamespace(
        shape=shape,
        dtype=dtype,
        device=device,
        is_contiguous=lambda: contiguous,
    )


def make_gating_inputs(tokens=17, heads=64, dtype=torch.bfloat16, head_dtype=torch.bfloat16):
    a = make_tensor((tokens, heads), dtype=dtype)
    b = make_tensor((tokens, heads), dtype=dtype)
    A_log = make_tensor((heads,), dtype=head_dtype)
    dt_bias = make_tensor((heads,), dtype=head_dtype)
    return A_log, a, b, dt_bias


def make_packed_qkv(tokens, q_dim, k_dim, v_dim):
    return torch.arange(tokens * (q_dim + k_dim + v_dim), dtype=torch.float32).to(torch.bfloat16)


@pytest.mark.parametrize("impl", [rearrange_qkv_module.IMPL_ASCENDC, rearrange_qkv_module.IMPL_TRITON])
def test_fused_path_returns_five_tensors(impl):
    tokens, q_dim, k_dim, v_dim, heads = 17, 1024, 1024, 3072, 64
    layer = make_layer()
    mixed_qkv = make_tensor((tokens, q_dim + k_dim + v_dim))
    A_log, a, b, dt_bias = make_gating_inputs(tokens=tokens, heads=heads)
    packed_qkv = make_packed_qkv(tokens, q_dim, k_dim, v_dim)
    g = torch.zeros(tokens, heads, dtype=torch.float32)
    beta = torch.zeros(tokens, heads, dtype=torch.bfloat16)

    custom_op = Mock(return_value=(packed_qkv, g, beta))
    triton_patch = Mock(return_value=(packed_qkv, g, beta))

    with (
        patch.object(rearrange_qkv_module, "_FUSED_IMPL_CACHE", impl),
        patch.object(torch.ops, "_C_ascend", SimpleNamespace(npu_rearrange_qkv_and_gdn_gating=custom_op)),
        patch("vllm_ascend.ops.triton.rearrange_qkv_gdn_gating.rearrange_qkv_and_gdn_gating_patch", triton_patch),
    ):
        query, key, value, g_out, beta_out = rearrange_qkv_module.rearrange_mixed_qkv_and_fused_gdn_gating(
            layer, mixed_qkv, A_log, a, b, dt_bias
        )

    if impl == rearrange_qkv_module.IMPL_ASCENDC:
        custom_op.assert_called_once_with(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            q_dim,
            k_dim,
            v_dim,
            rearrange_qkv_module.GDN_GATING_BETA,
            rearrange_qkv_module.GDN_GATING_THRESHOLD,
        )
        triton_patch.assert_not_called()
    else:
        triton_patch.assert_called_once()
        custom_op.assert_not_called()
    layer.rearrange_mixed_qkv.assert_not_called()

    assert query.shape == (1, tokens, q_dim // 128, 128)
    assert key.shape == (1, tokens, k_dim // 128, 128)
    assert value.shape == (1, tokens, v_dim // 128, 128)
    assert g_out.shape == (1, tokens, heads)
    assert beta_out.shape == (1, tokens, heads)
    assert torch.equal(query.reshape(tokens, -1), packed_qkv[: tokens * q_dim].view(tokens, q_dim))


@pytest.mark.parametrize(
    ("layer_attributes", "tokens", "heads", "dtype", "head_dtype", "contiguous"),
    [
        ({}, 17, 0, torch.bfloat16, torch.bfloat16, True),  # no heads to gate
        ({}, 17, 8192, torch.bfloat16, torch.bfloat16, True),  # head count above the tile cap
        ({}, 17, 64, torch.float16, torch.bfloat16, True),  # a is not bf16
        ({}, 17, 64, torch.bfloat16, torch.float32, True),  # A_log dtype mismatch
        ({}, 16, 64, torch.bfloat16, torch.bfloat16, False),  # non contiguous
        ({"key_dim": 2032}, 17, 64, torch.bfloat16, torch.bfloat16, True),  # misaligned q dim
    ],
)
def test_unsupported_layout_uses_unfused_path(layer_attributes, tokens, heads, dtype, head_dtype, contiguous):
    layer = make_layer(**layer_attributes)
    mixed_qkv = make_tensor((tokens, 2048 if not layer_attributes else 5104), contiguous=contiguous)
    A_log, a, b, dt_bias = make_gating_inputs(
        tokens=tokens, heads=heads, dtype=dtype, head_dtype=head_dtype
    )
    gating = Mock(return_value=(None, None))

    with (
        patch.object(rearrange_qkv_module, "_FUSED_IMPL_CACHE", rearrange_qkv_module.IMPL_ASCENDC),
        patch.object(rearrange_qkv_module, "fused_gdn_gating", gating),
    ):
        result = rearrange_qkv_module.rearrange_mixed_qkv_and_fused_gdn_gating(
            layer, mixed_qkv, A_log, a, b, dt_bias
        )

    assert result == (None, None, None, None, None)
    layer.rearrange_mixed_qkv.assert_called_once_with(mixed_qkv)
    gating.assert_called_once_with(A_log, a, b, dt_bias)


def test_gating_rows_must_match_rearrange_rows():
    layer = make_layer()
    mixed_qkv = make_tensor((17, 5120))
    A_log, a, b, dt_bias = make_gating_inputs(tokens=16, heads=64)
    gating = Mock(return_value=(None, None))

    with (
        patch.object(rearrange_qkv_module, "_FUSED_IMPL_CACHE", rearrange_qkv_module.IMPL_ASCENDC),
        patch.object(rearrange_qkv_module, "fused_gdn_gating", gating),
    ):
        result = rearrange_qkv_module.rearrange_mixed_qkv_and_fused_gdn_gating(
            layer, mixed_qkv, A_log, a, b, dt_bias
        )

    assert result == (None, None, None, None, None)
    layer.rearrange_mixed_qkv.assert_called_once_with(mixed_qkv)
    gating.assert_called_once_with(A_log, a, b, dt_bias)


def test_none_input_still_computes_gating():
    layer = make_layer()
    A_log, a, b, dt_bias = make_gating_inputs()
    gating = Mock(return_value=(Mock(), Mock()))

    with patch.object(rearrange_qkv_module, "fused_gdn_gating", gating):
        query, key, value, g, beta = rearrange_qkv_module.rearrange_mixed_qkv_and_fused_gdn_gating(
            layer, None, A_log, a, b, dt_bias
        )

    assert (query, key, value) == (None, None, None)
    assert (g, beta) == tuple(gating.return_value)
    layer.rearrange_mixed_qkv.assert_called_once_with(None)


def test_fused_impl_env_override_disables_fusion(monkeypatch):
    monkeypatch.setenv(rearrange_qkv_module.FUSED_IMPL_ENV, rearrange_qkv_module.IMPL_OFF)
    monkeypatch.setattr(rearrange_qkv_module, "_FUSED_IMPL_CACHE", False)
    assert rearrange_qkv_module._fused_impl() is None


def test_rearrange_only_path_is_unchanged():
    tokens, q_dim, k_dim, v_dim = 17, 1024, 1024, 3072
    layer = make_layer()
    mixed_qkv = SimpleNamespace(
        dtype=torch.bfloat16, shape=(tokens, q_dim + k_dim + v_dim), is_contiguous=lambda: True
    )
    packed_qkv = make_packed_qkv(tokens, q_dim, k_dim, v_dim)
    custom_op = Mock(return_value=packed_qkv)

    with (
        patch.object(rearrange_qkv_module, "SUPPORTS_REARRANGE_QKV", True),
        patch.object(torch.ops, "_C_ascend", SimpleNamespace(npu_rearrange_qkv=custom_op)),
    ):
        query, key, value = rearrange_qkv_module.rearrange_mixed_qkv(layer, mixed_qkv)

    custom_op.assert_called_once_with(mixed_qkv, q_dim, k_dim, v_dim)
    layer.rearrange_mixed_qkv.assert_not_called()
    assert query.shape == (1, tokens, q_dim // 128, 128)
    assert key.shape == (1, tokens, k_dim // 128, 128)
    assert value.shape == (1, tokens, v_dim // 128, 128)


def stable_softplus_gating(A_log, a, b, dt_bias, beta=1.0):
    """Runs on CPU only: mirrors the formula used by the AscendC/Triton kernels."""
    x = a.to(torch.float32) + dt_bias.to(torch.float32).unsqueeze(0)
    u = beta * x
    # softplus(u) = max(u, 0) + log(1 + exp(-|u|)), overflow free.
    softplus = torch.clamp(u, min=0.0) + torch.log1p(torch.exp(-u.abs()))
    g = -torch.exp(A_log.to(torch.float32)).unsqueeze(0) * softplus / beta
    # The kernels keep the gating math in fp32 and store beta back in b's dtype.
    beta_output = torch.sigmoid(b.to(torch.float32)).to(b.dtype)
    return g, beta_output


@pytest.mark.parametrize(("num_tokens", "num_heads"), [(1, 32), (17, 24), (129, 2), (1024, 64)])
@pytest.mark.parametrize("beta", [1.0, 0.5])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_kernel_formula_matches_reference_math(num_tokens, num_heads, beta, input_dtype):
    """The stable softplus used by both kernels must match the reference gating.

    Extreme inputs are included on purpose: the reference falls back to ``x``
    above the softplus threshold while the kernels rely on ``max(u, 0)``.
    """
    torch.manual_seed(1234)
    A_log = torch.rand(num_heads, dtype=torch.float32) * 4.0
    dt_bias = torch.randn(num_heads, dtype=torch.float32) * 2.0
    a = torch.cat(
        [
            torch.randn(num_tokens - 1, num_heads) * 5.0,
            torch.full((1, num_heads), 200.0),
        ]
    ).to(input_dtype)
    b = torch.randn(num_tokens, num_heads, dtype=torch.float32).to(input_dtype)

    ref_g, ref_beta = fused_gdn_gating_pytorch(
        A_log=A_log, a=a, b=b, dt_bias=dt_bias, beta=beta
    )
    g, beta_output = stable_softplus_gating(A_log, a, b, dt_bias, beta=beta)

    torch.testing.assert_close(g, ref_g.squeeze(0), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(beta_output, ref_beta.squeeze(0), rtol=1e-6, atol=1e-6)
