# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vllm-ascend project

import contextlib
import os

import torch

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

# One 32-byte DMA block contains 16 BF16 elements.
DMA_ALIGNMENT_ELEMENTS = 16
SUPPORTS_REARRANGE_QKV = get_ascend_device_type() in (AscendDeviceType.A2, AscendDeviceType.A3)

# Defaults of the reference gating implementation (`fused_gdn_gating`).
GDN_GATING_BETA = 1.0
GDN_GATING_THRESHOLD = 20.0
# Mirrors MAX_GATING_TILE_ELEMENTS in the AscendC tiling: one gating tile
# handles at most this many heads in a row.
MAX_GATING_HEADS = 4096

IMPL_AUTO = "auto"
IMPL_ASCENDC = "ascendc"
IMPL_TRITON = "triton"
IMPL_OFF = "off"
FUSED_IMPL_ENV = "VLLM_ASCEND_FUSED_QKV_GATING_IMPL"

_FUSED_IMPL_CACHE: str | None | bool = False


def _fused_dims(layer, mixed_qkv: torch.Tensor | None):
    """Return (q_dim, k_dim, v_dim) when the custom layout applies."""
    if (
        mixed_qkv is None
        or not SUPPORTS_REARRANGE_QKV
        or mixed_qkv.dtype != torch.bfloat16
        or not mixed_qkv.is_contiguous()
    ):
        return None

    q_dim = layer.key_dim // layer.tp_size
    k_dim = q_dim
    v_dim = layer.value_dim // layer.tp_size
    if q_dim % DMA_ALIGNMENT_ELEMENTS != 0 or v_dim % DMA_ALIGNMENT_ELEMENTS != 0:
        return None
    return q_dim, k_dim, v_dim


def _custom_op_available() -> bool:
    if not SUPPORTS_REARRANGE_QKV:
        return False
    try:
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            return False
        return hasattr(torch.ops._C_ascend, "npu_rearrange_qkv_and_gdn_gating")
    except Exception:  # pragma: no cover - depends on the runtime environment
        return False


def _triton_fused_available() -> bool:
    if not SUPPORTS_REARRANGE_QKV:
        return False
    try:
        from vllm_ascend.ops.triton.rearrange_qkv_gdn_gating import HAS_CORE_SCOPE

        return HAS_CORE_SCOPE
    except Exception:  # pragma: no cover - depends on the installed triton-ascend
        return False


def _fused_impl() -> str | None:
    """Resolve which fused implementation to use, honoring the env override."""
    global _FUSED_IMPL_CACHE
    if _FUSED_IMPL_CACHE is not False:
        return _FUSED_IMPL_CACHE  # type: ignore[return-value]

    requested = os.getenv(FUSED_IMPL_ENV, IMPL_AUTO).strip().lower()
    resolved: str | None
    if requested in (IMPL_OFF, "none", "0", "false"):
        resolved = None
    elif requested == IMPL_ASCENDC:
        resolved = IMPL_ASCENDC if _custom_op_available() else None
    elif requested == IMPL_TRITON:
        resolved = IMPL_TRITON if _triton_fused_available() else None
    else:
        if _custom_op_available():
            resolved = IMPL_ASCENDC
        elif _triton_fused_available():
            resolved = IMPL_TRITON
        else:
            resolved = None
    _FUSED_IMPL_CACHE = resolved
    return resolved


@contextlib.contextmanager
def _impl_override(impl: str | None):
    """Force a fused implementation inside the context (benchmarks and tests)."""
    global _FUSED_IMPL_CACHE
    previous = _FUSED_IMPL_CACHE
    _FUSED_IMPL_CACHE = impl
    try:
        yield
    finally:
        _FUSED_IMPL_CACHE = previous


def _split_packed_qkv(packed_qkv: torch.Tensor, layer, num_tokens: int, q_dim: int, k_dim: int, v_dim: int):
    query, key, value = packed_qkv.split([num_tokens * q_dim, num_tokens * k_dim, num_tokens * v_dim])
    return (
        query.view(1, num_tokens, q_dim // layer.head_k_dim, layer.head_k_dim),
        key.view(1, num_tokens, k_dim // layer.head_k_dim, layer.head_k_dim),
        value.view(1, num_tokens, v_dim // layer.head_v_dim, layer.head_v_dim),
    )


def _gating_inputs_supported(impl: str | None, mixed_qkv: torch.Tensor | None, A_log, a, b, dt_bias) -> bool:
    if impl is None or mixed_qkv is None:
        return False
    if a.shape[0] != mixed_qkv.shape[0] or a.shape != b.shape:
        return False
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        return False
    if not (a.is_contiguous() and b.is_contiguous() and A_log.is_contiguous() and dt_bias.is_contiguous()):
        return False
    if A_log.dtype != dt_bias.dtype or not A_log.dtype.is_floating_point:
        return False
    if a.device != mixed_qkv.device or A_log.device != mixed_qkv.device:
        return False
    if impl == IMPL_ASCENDC:
        num_heads = a.shape[1]
        # The kernel handles unaligned head counts with padded copies, but the
        # UB tile is sized for at most MAX_GATING_TILE_ELEMENTS elements.
        if num_heads == 0 or num_heads > MAX_GATING_HEADS:
            return False
        if A_log.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            return False
    return True


def fused_gdn_gating(A_log, a: torch.Tensor, b: torch.Tensor, dt_bias):
    """Gating via the device adaptor, imported lazily to avoid import cycles."""
    from vllm_ascend.device.device_op import DeviceOperator

    return DeviceOperator.fused_gdn_gating(A_log, a, b, dt_bias)


def rearrange_mixed_qkv(layer, mixed_qkv: torch.Tensor | None):
    """Use the custom QKV rearrange when the device and layout support it."""
    dims = _fused_dims(layer, mixed_qkv)
    if dims is None:
        return layer.rearrange_mixed_qkv(mixed_qkv)

    num_tokens = mixed_qkv.shape[0]
    packed_qkv = torch.ops._C_ascend.npu_rearrange_qkv(mixed_qkv, *dims)
    return _split_packed_qkv(packed_qkv, layer, num_tokens, *dims)


def rearrange_mixed_qkv_and_fused_gdn_gating(
    layer,
    mixed_qkv: torch.Tensor | None,
    A_log,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias,
):
    """Run the QKV rearrange and the gating in one kernel when possible.

    The fused kernels keep the DMA rearrange on the cube cores and the gating
    on the vector cores so that both run in parallel. The gating outputs are
    always returned, the Q/K/V outputs are ``None`` when ``mixed_qkv`` is.

    Returns:
        (query, key, value, g, beta_output)
    """
    dims = _fused_dims(layer, mixed_qkv)
    impl = _fused_impl()
    if not _gating_inputs_supported(impl, mixed_qkv, A_log, a, b, dt_bias):
        query, key, value = layer.rearrange_mixed_qkv(mixed_qkv)
        g, beta_output = fused_gdn_gating(A_log, a, b, dt_bias)
        return query, key, value, g, beta_output

    assert dims is not None and mixed_qkv is not None
    q_dim, k_dim, v_dim = dims
    num_tokens = mixed_qkv.shape[0]
    if impl == IMPL_ASCENDC:
        packed_qkv, g, beta_output = torch.ops._C_ascend.npu_rearrange_qkv_and_gdn_gating(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            q_dim,
            k_dim,
            v_dim,
            GDN_GATING_BETA,
            GDN_GATING_THRESHOLD,
        )
    else:
        from vllm_ascend.ops.triton.rearrange_qkv_gdn_gating import rearrange_qkv_and_gdn_gating_patch

        packed_qkv, g, beta_output = rearrange_qkv_and_gdn_gating_patch(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            q_dim,
            k_dim,
            v_dim,
            beta=GDN_GATING_BETA,
            threshold=GDN_GATING_THRESHOLD,
        )

    query, key, value = _split_packed_qkv(packed_qkv, layer, num_tokens, q_dim, k_dim, v_dim)
    return query, key, value, g.view(1, num_tokens, -1), beta_output.view(1, num_tokens, -1)
