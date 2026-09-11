# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

"""Fused GDN prologue for Triton-Ascend.

The kernel mirrors `csrc/attention/rearrange_qkv_gdn_gating`: the cube cores
pack the token major mixed QKV into the [query | key | value] blocks with plain
MTE (DMA) moves while the vector cores compute the gating in parallel.

`triton.language.extra.cann.extension.scope` is what lets a single Triton kernel
declare one cube scope and one vector scope; both scopes run concurrently and
`sub_vec_id` splits the two vector cores that belong to one cube program.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_aicore_num

try:  # pragma: no cover - depends on the installed triton-ascend build
    import triton.language.extra.cann.extension as _cann_extension
except ImportError:  # pragma: no cover
    _cann_extension = None

HAS_CORE_SCOPE = _cann_extension is not None and hasattr(_cann_extension, "scope")
HAS_SUB_VEC_ID = _cann_extension is not None and hasattr(_cann_extension, "sub_vec_id")

# Every cube core is paired with two vector cores on A2/A3.
VECTOR_CORES_PER_CUBE = 2
# Keep the same launch cap as the AscendC tiling.
MAX_PROGRAMS = 20
COPY_BLOCK_ROWS = 8
COPY_BLOCK_COLS = 512
GATING_BLOCK_ROWS = 64
GATING_BLOCK_HEADS = 64


@triton.jit
def _copy_qkv_segment(
    x_ptr,
    y_ptr,
    row_offs,
    row_mask,
    src_col,
    dst_col,
    seg_dim,
    row_dim,
    COPY_ROWS: tl.constexpr,
    COPY_COLS: tl.constexpr,
):
    """Copy one [query | key | value] segment of the rows into its own block."""
    for col_start in range(0, seg_dim, COPY_COLS):
        cols = col_start + tl.arange(0, COPY_COLS)
        mask = row_mask[:, None] & (cols[None, :] < seg_dim)
        x_offs = row_offs[:, None] * row_dim + src_col + cols[None, :]
        y_offs = row_offs[:, None] * seg_dim + dst_col + cols[None, :]
        values = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
        tl.store(y_ptr + y_offs, values, mask=mask)


@triton.jit
def rearrange_qkv_gdn_gating_kernel(
    x_ptr,
    a_ptr,
    b_ptr,
    a_log_ptr,
    dt_bias_ptr,
    y_ptr,
    g_ptr,
    beta_ptr,
    num_tokens,
    q_dim,
    k_dim,
    v_dim,
    row_dim,
    num_heads,
    num_cube_programs,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    COPY_ROWS: tl.constexpr,
    COPY_COLS: tl.constexpr,
    GATE_ROWS: tl.constexpr,
    GATE_HEADS: tl.constexpr,
    VEC_PER_CUBE: tl.constexpr,
    USE_SUB_VEC_ID: tl.constexpr,
):
    # ------------------------------------------------------------------
    # Cube cores: rearrange the mixed QKV with DMA moves only.
    # ------------------------------------------------------------------
    with _cann_extension.scope(core_mode="cube"):
        cube_pid = tl.program_id(0)
        rows_per_program = tl.cdiv(num_tokens, num_cube_programs)
        row_begin = cube_pid * rows_per_program
        row_count = tl.minimum(rows_per_program, num_tokens - row_begin)
        for row_start in range(0, row_count, COPY_ROWS):
            row_offs = row_begin + row_start + tl.arange(0, COPY_ROWS)
            row_mask = row_offs < row_begin + row_count
            _copy_qkv_segment(
                x_ptr,
                y_ptr,
                row_offs,
                row_mask,
                0,
                0,
                q_dim,
                row_dim,
                COPY_ROWS,
                COPY_COLS,
            )
            _copy_qkv_segment(
                x_ptr,
                y_ptr,
                row_offs,
                row_mask,
                q_dim,
                num_tokens * q_dim,
                k_dim,
                row_dim,
                COPY_ROWS,
                COPY_COLS,
            )
            _copy_qkv_segment(
                x_ptr,
                y_ptr,
                row_offs,
                row_mask,
                q_dim + k_dim,
                num_tokens * (q_dim + k_dim),
                v_dim,
                row_dim,
                COPY_ROWS,
                COPY_COLS,
            )

    # ------------------------------------------------------------------
    # Vector cores: gating, executed in parallel with the cube scope.
    # ------------------------------------------------------------------
    with _cann_extension.scope(core_mode="vector"):
        vector_pid = tl.program_id(0)
        num_programs = tl.num_programs(0)
        worker = vector_pid
        workers = num_programs
        if USE_SUB_VEC_ID:
            if num_programs == num_cube_programs:
                # The vector scope inherited the cube grid: spread the two
                # vector cores of every cube program over two row ranges.
                # sub_vec_id() is int64, cast it back to keep the row index types
                # identical in both branches.
                worker = vector_pid * VEC_PER_CUBE + _cann_extension.sub_vec_id().to(tl.int32)
                workers = num_cube_programs * VEC_PER_CUBE

        rows_per_worker = tl.cdiv(num_tokens, workers)
        row_begin = worker * rows_per_worker
        row_count = tl.minimum(rows_per_worker, num_tokens - row_begin)
        for row_start in range(0, row_count, GATE_ROWS):
            row_offs = row_begin + row_start + tl.arange(0, GATE_ROWS)
            row_mask = row_offs < row_begin + row_count
            for head_start in range(0, num_heads, GATE_HEADS):
                heads = head_start + tl.arange(0, GATE_HEADS)
                head_mask = heads < num_heads
                mask = row_mask[:, None] & head_mask[None, :]
                offs = row_offs[:, None] * num_heads + heads[None, :]

                dt_bias = tl.load(dt_bias_ptr + heads, mask=head_mask, other=0.0).to(tl.float32)
                a_log = tl.load(a_log_ptr + heads, mask=head_mask, other=0.0).to(tl.float32)
                a_value = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                b_value = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

                x = a_value + dt_bias[None, :]
                u = BETA * x
                # Numerically stable softplus: log(1 + exp(u)) without overflow.
                softplus = tl.maximum(u, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(u)))
                g = -tl.exp(a_log)[None, :] * softplus / BETA
                tl.store(g_ptr + offs, g, mask=mask)

                beta_value = 1.0 / (1.0 + tl.exp(-b_value))
                tl.store(beta_ptr + offs, beta_value.to(beta_ptr.dtype.element_ty), mask=mask)


def rearrange_qkv_and_gdn_gating_patch(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused rearrange + gating kernel.

    Args:
        mixed_qkv: [num_tokens, q_dim + k_dim + v_dim] bf16 conv1d output.
        a: [num_tokens, num_heads] gate input.
        b: [num_tokens, num_heads] beta input.
        A_log: [num_heads] log decay parameter.
        dt_bias: [num_heads] time step bias.

    Returns:
        The packed QKV buffer, g in fp32 and beta in the dtype of ``b``, both
        laid out as [num_tokens, num_heads].
    """
    if not HAS_CORE_SCOPE:
        raise RuntimeError(
            "the installed triton-ascend does not provide scoped cube/vector execution; "
            "update triton-ascend or select the AscendC implementation"
        )

    num_tokens, row_dim = mixed_qkv.shape
    num_heads = a.shape[1]
    num_cube_programs = max(1, min(get_aicore_num(), MAX_PROGRAMS, num_tokens))

    packed_qkv = torch.empty(num_tokens * row_dim, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    g = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=mixed_qkv.device)
    beta_output = torch.empty((num_tokens, num_heads), dtype=mixed_qkv.dtype, device=mixed_qkv.device)

    grid = (num_cube_programs,)
    rearrange_qkv_gdn_gating_kernel[grid](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        packed_qkv,
        g,
        beta_output,
        num_tokens,
        q_dim,
        k_dim,
        v_dim,
        row_dim,
        num_heads,
        num_cube_programs,
        BETA=beta,
        THRESHOLD=threshold,
        COPY_ROWS=COPY_BLOCK_ROWS,
        COPY_COLS=COPY_BLOCK_COLS,
        GATE_ROWS=GATING_BLOCK_ROWS,
        GATE_HEADS=GATING_BLOCK_HEADS,
        VEC_PER_CUBE=VECTOR_CORES_PER_CUBE,
        USE_SUB_VEC_ID=HAS_SUB_VEC_ID,
    )
    return packed_qkv, g, beta_output
