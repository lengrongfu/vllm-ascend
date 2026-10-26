# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd_head_first
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 15, 17, 127, 129])
@pytest.mark.parametrize("layout", ["sequence", "head", "slice", "strided_dim"])
def test_prefill_l2norm_head_first(dtype, tokens, layout):
    init_device_properties_triton()
    torch.manual_seed(42)
    x = torch.randn(2, tokens + 4, 4, 128, device="npu", dtype=dtype)
    if layout == "head":
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    elif layout == "strided_dim":
        x = torch.randn(2, tokens + 4, 4, 256, device="npu", dtype=dtype)[..., ::2]
    x = x[:, 2 : tokens + 2]
    if layout == "sequence":
        x = x.contiguous()
    values = x.float()
    expected = (values * torch.rsqrt(values.square().sum(-1, keepdim=True) + 1e-6)).to(dtype)
    actual = l2norm_fwd_head_first(x)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected.transpose(1, 2), rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("shape", [(0, 3, 4, 128), (1, 0, 4, 128), (1, 3, 0, 128)])
def test_prefill_l2norm_empty(shape):
    result = l2norm_fwd_head_first(torch.empty(shape, device="npu"))
    assert result.shape == (shape[0], shape[2], shape[1], shape[3])
    assert result.numel() == 0


def test_prefill_l2norm_zero_rows():
    init_device_properties_triton()
    x = torch.zeros(1, 17, 4, 128, device="npu", dtype=torch.bfloat16)
    assert torch.count_nonzero(l2norm_fwd_head_first(x)).item() == 0


def test_prefill_l2norm_peak_memory():
    # The old norm -> transpose path keeps both full-size Q/K buffers alive.
    # Verify the fused operation needs only its head-major output allocation.
    init_device_properties_triton()
    x = torch.randn(1, 8192, 4, 128, device="npu", dtype=torch.bfloat16)
    baseline = lambda: l2norm_fwd(x).transpose(1, 2).contiguous()
    optimized = lambda: l2norm_fwd_head_first(x)
    for operation in (baseline, optimized):
        operation()
    torch.npu.synchronize()

    def peak_allocation(operation):
        torch.npu.reset_peak_memory_stats()
        before = torch.npu.memory_allocated()
        result = operation()
        torch.npu.synchronize()
        peak = torch.npu.max_memory_allocated() - before
        del result
        return peak

    assert peak_allocation(optimized) < peak_allocation(baseline)


@pytest.mark.parametrize("dim", [63, 192, 256, 1024])
def test_prefill_l2norm_feature_tail(dim):
    init_device_properties_triton()
    x = torch.randn(1, 17, 2, dim, device="npu", dtype=torch.bfloat16)
    values = x.float()
    expected = (values * torch.rsqrt(values.square().sum(-1, keepdim=True) + 1e-6)).to(x.dtype).transpose(1, 2)
    torch.testing.assert_close(l2norm_fwd_head_first(x), expected, rtol=1e-3, atol=1e-6)
