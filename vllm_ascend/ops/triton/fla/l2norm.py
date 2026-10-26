# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fla/ops/l2norm.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["eps", "M", "NUM_CHUNKS"])
def l2norm_fwd_kernel2_loop(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr, NUM_CHUNKS):
    base_row = tl.program_id(0) * (NUM_CHUNKS * MBLOCK)
    rindex = tl.arange(0, N)[None, :]

    for chunk in range(NUM_CHUNKS):
        row_idx = base_row + chunk * MBLOCK + tl.arange(0, MBLOCK)[:, None]
        xmask = row_idx < M

        xs = tl.load(X + (rindex + N * row_idx), mask=xmask, other=0.0).to(tl.float32)
        square = xs * xs
        square_sum = tl.sum(square, 1)[:, None]
        rsqrt = tl.rsqrt(square_sum + eps)

        tl.store(Y + (rindex + N * row_idx), xs * rsqrt, xmask)


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError(f"l2norm_fwd: This layer doesn't support feature dim >= 64KB, got {D}.")

    MBLOCK = 69
    # M, N = x.shape
    num_core = get_vectorcore_num()
    main_bs = triton.cdiv(T, num_core)
    num_sub_blocks = triton.cdiv(main_bs, MBLOCK)
    grid = (num_core,)
    l2norm_fwd_kernel2_loop[grid](
        X=x,
        Y=y,
        eps=eps,
        M=T,
        N=D,
        MBLOCK=MBLOCK,
        NUM_CHUNKS=num_sub_blocks,
    )

    return y.view(x_shape_og)


_HEAD_FIRST_L2NORM_MAX_DIM = 1024
_HEAD_FIRST_L2NORM_ROW_ELEMENTS = 2048
_HEAD_FIRST_L2NORM_STRIDED_ROW_ELEMENTS = 512


@triton.jit
def _l2norm_head_first_kernel(
    X,
    Y,
    eps,
    rows,
    tokens,
    heads,
    stride_batch,
    stride_token,
    stride_head,
    stride_dim,
    DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    columns = tl.arange(0, BLOCK_DIM)[None, :]
    for block in range(tl.program_id(0), tl.cdiv(rows, BLOCK_ROWS), tl.num_programs(0)):
        row = block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        token = row % tokens
        head = (row // tokens) % heads
        batch = row // (tokens * heads)
        offsets = batch * stride_batch + token * stride_token + head * stride_head + columns * stride_dim
        mask = (row < rows) & (columns < DIM)
        values = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        inverse_norm = tl.rsqrt(tl.sum(values * values, axis=1)[:, None] + eps)
        tl.store(Y + row * DIM + columns, values * inverse_norm, mask=mask)


def l2norm_fwd_head_first(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize strided BSND input directly into contiguous BNSD output.

    FP32 reduction and the epsilon match ``l2norm_fwd``. Reading the original
    strides avoids packing Q/K before normalization and transposing its output.
    """
    if x.ndim != 4:
        raise ValueError("head-first L2Norm requires a 4D BSND tensor")
    batch, tokens, heads, dim = x.shape
    if dim == 0:
        raise ValueError("head-first L2Norm requires a nonzero feature dimension")
    if dim > _HEAD_FIRST_L2NORM_MAX_DIM:
        # Preserve the existing implementation for unusually wide heads.
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd as upstream_l2norm_fwd

        return upstream_l2norm_fwd(x.contiguous(), eps=eps).transpose(1, 2).contiguous()
    output = torch.empty((batch, heads, tokens, dim), dtype=x.dtype, device=x.device)
    rows = batch * tokens * heads
    if rows == 0:
        return output
    block_dim = triton.next_power_of_2(dim)
    row_elements = _HEAD_FIRST_L2NORM_ROW_ELEMENTS if x.stride(-1) == 1 else _HEAD_FIRST_L2NORM_STRIDED_ROW_ELEMENTS
    block_rows = max(1, row_elements // block_dim)
    grid = (min(get_vectorcore_num(), triton.cdiv(rows, block_rows)),)
    _l2norm_head_first_kernel[grid](
        x,
        output,
        eps,
        rows,
        tokens,
        heads,
        *x.stride(),
        DIM=dim,
        BLOCK_DIM=block_dim,
        BLOCK_ROWS=block_rows,
        num_warps=4,
        num_stages=1,
        multibuffer=False,
    )
    return output
