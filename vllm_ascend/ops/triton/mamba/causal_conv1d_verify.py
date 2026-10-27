# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Linear speculative Conv1D with FLA-compatible compact state writeback.

The next round selects three history tokens at ``num_accepted_tokens - 1``.
Write each input while it is in registers, avoiding a second input read and
full [steps, history, channels] snapshots.
"""

import torch
from vllm.triton_utils import tl, triton

_CONV_WIDTH = 4
_HISTORY_TOKENS = _CONV_WIDTH - 1
_MAX_VERIFY_TOKENS = 8
_MAX_VERIFY_REQUESTS = 32
_CHANNEL_BLOCK = 256


@triton.jit
def _causal_conv1d_verify_kernel(
    x_ptr,
    state_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    slots_ptr,
    query_start_ptr,
    accepted_ptr,
    CHANNELS: tl.constexpr,
    STATE_SLOTS: tl.constexpr,
    STATE_LENGTH: tl.constexpr,
    STATE_SLOT_STRIDE: tl.constexpr,
    STATE_TIME_STRIDE: tl.constexpr,
    STATE_CHANNEL_STRIDE: tl.constexpr,
    INPUT_TIME_STRIDE: tl.constexpr,
    INPUT_CHANNEL_STRIDE: tl.constexpr,
    WEIGHT_TIME_STRIDE: tl.constexpr,
    WEIGHT_CHANNEL_STRIDE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots_ptr + request).to(tl.int64)
    start = tl.load(query_start_ptr + request)
    end = tl.load(query_start_ptr + request + 1)
    if (slot > 0) & (slot < STATE_SLOTS) & (end > start):
        mask = channel < CHANNELS
        accepted = tl.load(accepted_ptr + request)
        offset = tl.minimum(tl.maximum(accepted - 1, 0), STATE_LENGTH - 3)
        state = state_ptr + slot * STATE_SLOT_STRIDE + channel * STATE_CHANNEL_STRIDE
        history0 = tl.load(state + offset * STATE_TIME_STRIDE, mask, 0).to(tl.float32)
        history1 = tl.load(state + (offset + 1) * STATE_TIME_STRIDE, mask, 0).to(tl.float32)
        history2 = tl.load(state + (offset + 2) * STATE_TIME_STRIDE, mask, 0).to(tl.float32)
        weight = weight_ptr + channel * WEIGHT_CHANNEL_STRIDE
        weight0 = tl.load(weight, mask, 0).to(tl.float32)
        weight1 = tl.load(weight + WEIGHT_TIME_STRIDE, mask, 0).to(tl.float32)
        weight2 = tl.load(weight + 2 * WEIGHT_TIME_STRIDE, mask, 0).to(tl.float32)
        weight3 = tl.load(weight + 3 * WEIGHT_TIME_STRIDE, mask, 0).to(tl.float32)
        bias = tl.full((BLOCK,), 0, tl.float32)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + channel, mask, 0).to(tl.float32)

        # Load overlapping history before stores. Requests own distinct slots;
        # channel tiles access disjoint elements, including padded state slots.
        tl.store(state, history1, mask)
        tl.store(state + STATE_TIME_STRIDE, history2, mask)
        for token in range(start, end):
            current = tl.load(x_ptr + token * INPUT_TIME_STRIDE + channel * INPUT_CHANNEL_STRIDE, mask, 0).to(
                tl.float32
            )
            result = history0 * weight0 + history1 * weight1 + history2 * weight2 + current * weight3 + bias
            if SILU:
                result *= tl.sigmoid(result)
            tl.store(out_ptr + token * CHANNELS + channel, result, mask)
            tl.store(state + (2 + token - start) * STATE_TIME_STRIDE, current, mask)
            history0 = history1
            history1 = history2
            history2 = current
    else:
        # Do not touch null slots or leak uninitialized graph-padding output.
        for token in range(start, end):
            tl.store(out_ptr + token * CHANNELS + channel, 0, channel < CHANNELS)


def causal_conv1d_verify(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    conv_state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    max_query_len: int,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> torch.Tensor | None:
    """Try short width-four verification, returning None for FLA fallback.

    Inputs: x[tokens, channels], weight[4, channels], state[pool, window,
    channels]. Slots must be distinct for live requests; slot zero is null.
    Acceptance and sequence metadata remain on device. Supports padded slots
    and strided inputs/weights without contiguous copies. max_query_len must
    bound the device-resident query lengths, as in the FLA interface.
    """
    if (
        x.ndim != 2
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.device.type != "npu"
        or weight.shape != (_CONV_WIDTH, x.shape[1])
        or conv_state.ndim != 3
        or conv_state.shape[2] != x.shape[1]
        or not 1 <= max_query_len <= _MAX_VERIFY_TOKENS
        or conv_state.shape[1] < max_query_len + _HISTORY_TOKENS - 1
        or activation not in (None, "silu", "swish")
        or conv_state.stride(2) != 1
        or conv_state.stride(1) < x.shape[1]
        or conv_state.stride(0) < conv_state.shape[1] * conv_state.stride(1)
        or conv_state_indices.ndim != 1
        or conv_state_indices.numel() > _MAX_VERIFY_REQUESTS
        or query_start_loc.ndim != 1
        or num_accepted_tokens.ndim != 1
        or query_start_loc.numel() != conv_state_indices.numel() + 1
        or num_accepted_tokens.numel() != conv_state_indices.numel()
    ):
        return None
    data = (conv_state, weight) if bias is None else (conv_state, weight, bias)
    if any(t.device != x.device or t.dtype != x.dtype for t in data):
        return None
    if bias is not None and (bias.shape != (x.shape[1],) or not bias.is_contiguous()):
        return None
    metadata = (conv_state_indices, query_start_loc, num_accepted_tokens)
    if any(
        t.device != x.device or t.dtype not in (torch.int32, torch.int64) or not t.is_contiguous() for t in metadata
    ):
        return None
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if x.numel() == 0 or conv_state_indices.numel() == 0:
        return output
    _causal_conv1d_verify_kernel[(conv_state_indices.numel(), triton.cdiv(x.shape[1], _CHANNEL_BLOCK))](
        x,
        conv_state,
        weight,
        bias,
        output,
        conv_state_indices,
        query_start_loc,
        num_accepted_tokens,
        x.shape[1],
        conv_state.shape[0],
        conv_state.shape[1],
        *conv_state.stride(),
        *x.stride(),
        *weight.stride(),
        bias is not None,
        activation is not None,
        _CHANNEL_BLOCK,
        enable_fp_fusion=False,
    )
    return output
