# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.mamba.causal_conv1d_verify import causal_conv1d_verify


def reference(x, state, weight, slots, starts, accepted, bias, activation):
    output = torch.zeros_like(x)
    for request, slot in enumerate(slots):
        start, end = starts[request : request + 2]
        if slot <= 0 or start == end:
            continue
        offset = min(max(accepted[request] - 1, 0), state.shape[1] - 3)
        history = state[slot, offset : offset + 3].clone()
        window = torch.cat((history, x[start:end]), dim=0)
        for t in range(end - start):
            result = (window[t : t + 4].float() * weight.float()).sum(0)
            if bias is not None:
                result += bias.float()
            if activation is not None:
                result = torch.nn.functional.silu(result)
            output[start + t] = result.to(x.dtype)
        state[slot, : end - start + 2] = window[1:]
    return output


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("channels", [80, 2048])
@pytest.mark.parametrize("steps", [1, 4, 8])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_verify_compact_window_rollback(dtype, channels, steps, strided, activation):
    torch.manual_seed(42)
    lengths = [steps, max(steps - 1, 0), 1, 0, steps]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    slots = [2, 4, 1, 5, 0]
    window_size = steps + 2
    # Padding between slots is the model runner's actual state-pool layout.
    storage = torch.randn(6, window_size + 2, channels, dtype=dtype) * 0.2
    state = storage[:, :window_size] if strided else storage[:, :window_size].contiguous()
    expected_state = state.clone()
    device_storage = storage.npu()
    device_state = device_storage[:, :window_size] if strided else state.npu()
    weight = torch.randn(4, channels, dtype=dtype) * 0.2
    bias = torch.randn(channels, dtype=dtype) * 0.2 if strided else None
    device_weight = weight.T.contiguous().T.npu() if strided else weight.npu()
    device_bias = None if bias is None else bias.npu()
    device_slots = torch.tensor(slots, device="npu", dtype=torch.int32)
    device_starts = torch.tensor(starts, device="npu", dtype=torch.int32)
    for accepted in ([1] * 5, [steps, 1, steps, 1, 1], [1, steps, 1, 1, 1]):
        raw = torch.randn(starts[-1], channels * (2 if strided else 1), dtype=dtype) * 0.2
        x = raw[:, ::2] if strided else raw
        device_x = raw.npu()[:, ::2] if strided else x.npu()
        expected = reference(x, expected_state, weight, slots, starts, accepted, bias, activation)
        actual = causal_conv1d_verify(
            device_x,
            device_state,
            device_weight,
            device_slots,
            device_starts,
            torch.tensor(accepted, device="npu", dtype=torch.int32),
            steps,
            device_bias,
            activation,
        )
        assert actual is not None
        torch.testing.assert_close(actual.cpu(), expected, atol=0.002, rtol=0.01)
        torch.testing.assert_close(device_state.cpu(), expected_state, atol=0, rtol=0)
        if strided:
            torch.testing.assert_close(device_storage[:, window_size:].cpu(), storage[:, window_size:], atol=0, rtol=0)


def test_verify_fallback_does_not_mutate_state():
    x = torch.zeros(4, 80, device="npu", dtype=torch.bfloat16)
    state = torch.randn(2, 6, 80, device="npu", dtype=x.dtype)
    original = state.clone()
    metadata = torch.ones(1, device="npu", dtype=torch.int32)
    starts = torch.tensor([0, 4], device="npu", dtype=torch.int32)
    for weight, max_query in [
        (torch.ones(3, 80, device="npu", dtype=x.dtype), 4),
        (torch.ones(4, 80, device="npu", dtype=x.dtype), 9),
    ]:
        assert causal_conv1d_verify(x, state, weight, metadata, starts, metadata, max_query) is None
    too_many = torch.ones(33, device="npu", dtype=torch.int32)
    padded_starts = torch.zeros(34, device="npu", dtype=torch.int32)
    assert (
        causal_conv1d_verify(
            x, state, torch.ones(4, 80, device="npu", dtype=x.dtype), too_many, padded_starts, too_many, 4
        )
        is None
    )
    torch.testing.assert_close(state, original, atol=0, rtol=0)


def test_verify_graph_replay_uses_updated_acceptance():
    torch.manual_seed(7)
    channels, steps = 80, 4
    state = torch.randn(3, steps + 2, channels, dtype=torch.bfloat16) * 0.2
    weight = torch.randn(4, channels, dtype=state.dtype) * 0.2
    x = torch.randn(steps, channels, dtype=state.dtype) * 0.2
    device_state, device_weight, device_x = state.npu(), weight.npu(), x.npu()
    slots = torch.tensor([1], device="npu", dtype=torch.int64)
    starts = torch.tensor([0, steps], device="npu", dtype=torch.int64)
    accepted = torch.ones(1, device="npu", dtype=torch.int64)
    causal_conv1d_verify(device_x, device_state, device_weight, slots, starts, accepted, steps)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = causal_conv1d_verify(device_x, device_state, device_weight, slots, starts, accepted, steps)
    assert actual is not None
    device_state.copy_(state)
    expected_state = state.clone()
    for count in [1, 4, 2]:
        accepted.fill_(count)
        graph.replay()
        expected = reference(x, expected_state, weight, [1], [0, steps], [count], None, "silu")
        torch.testing.assert_close(actual.cpu(), expected, atol=0.002, rtol=0.01)
        torch.testing.assert_close(device_state.cpu(), expected_state, atol=0, rtol=0)
