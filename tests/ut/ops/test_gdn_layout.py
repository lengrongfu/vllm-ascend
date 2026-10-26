# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention


@pytest.mark.parametrize("kind", ["head_first", "sequence_first", "mixed_slice"])
def test_a5_prefill_layout_preserves_values_and_output(kind):
    tokens, nk, nv, dim = 7, 2, 4, 16
    qkv = [torch.randn(1, h, tokens, dim).transpose(1, 2) for h in (nk, nk, nv)]
    if kind == "sequence_first":
        qkv = [x.contiguous() for x in qkv]
    elif kind == "mixed_slice":
        qkv = [x[:, 2:] for x in qkv]
    count = qkv[0].shape[1]
    state = torch.randn(1, nv, dim, dim)
    output = torch.randn(1, count, nv, dim)
    gate = torch.randn(1, count, nv)
    meta = SimpleNamespace(cu_seqlens_host=None, chunk_indices_chunk64_host=None, keep_meta=None)

    def fused(q, k, v, g, beta, **kwargs):
        expected_layout = "BNSD" if kind == "head_first" else "BSND"
        assert kwargs["layout"] == expected_layout
        for actual, original in zip((q, k, v), qkv):
            assert actual.is_contiguous()
            expected = original.transpose(1, 2) if kind == "head_first" else original
            torch.testing.assert_close(actual, expected)
            if kind == "head_first":
                assert actual.data_ptr() == original.data_ptr()
        torch.testing.assert_close(g, gate)
        torch.testing.assert_close(beta, gate)
        return output, state

    result, final_state = A5DeviceAdaptor.fla_gdn_prefill(*qkv, gate, gate, state, dim**-0.5, meta, fused)
    assert result is output
    assert final_state is state


@pytest.mark.parametrize("backend", ["base", "cann"])
@pytest.mark.parametrize("sliced", [False, True])
def test_head_first_l2norm_fallback_accepts_multiple_heads(backend, sliced):
    tokens, nk, nv, dim = 7, 2, 4, 128
    qkv = [torch.randn(1, heads, tokens, dim).transpose(1, 2) for heads in (nk, nk, nv)]
    if sliced:
        qkv = [x[:, :2] for x in qkv]
    count = qkv[0].shape[1]
    state = torch.randn(1, nv, dim, dim)
    gate = torch.zeros(1, count, nv)
    output = torch.randn(1, count, nv, dim)

    def norm(x):
        # Match the real upstream L2Norm's first operation. It raises on
        # head-first BSND views unless the caller packs them before norm.
        x.view(-1, x.shape[-1])
        return torch.nn.functional.normalize(x, dim=-1)

    if backend == "base":
        meta = SimpleNamespace(cu_seqlens_host=(0, count), chunk_indices_chunk64_host=((0, 0),), keep_meta=None)

        def fused(q, k, v, *args, **kwargs):
            assert kwargs["layout"] == "BNSD"
            for got, original in zip((q, k), qkv[:2]):
                expected = torch.nn.functional.normalize(original, dim=-1).transpose(1, 2)
                torch.testing.assert_close(got, expected)
            torch.testing.assert_close(v, qkv[2].transpose(1, 2))
            return output, state.transpose(-1, -2).contiguous()

        with patch(
            "vllm_ascend.ops.triton.fla.l2norm.l2norm_fwd_head_first",
            side_effect=lambda x: norm(x.contiguous()).transpose(1, 2).contiguous(),
        ):
            actual, _ = BaseDeviceAdaptor.fla_gdn_prefill(*qkv, gate, gate, state, dim**-0.5, meta, fused)
    else:

        def fused(q, k, v, *args, **kwargs):
            for got, original in zip((q, k), qkv[:2]):
                torch.testing.assert_close(got, torch.nn.functional.normalize(original, dim=-1).squeeze(0))
            torch.testing.assert_close(v, qkv[2].squeeze(0))
            return output.squeeze(0), kwargs["initial_state"]

        with (
            patch("vllm_ascend.ops.gdn.l2norm_fwd", side_effect=norm),
            patch("vllm_ascend.ops.gdn.torch_npu.npu_chunk_gated_delta_rule", side_effect=fused, create=True),
        ):
            actual, _ = AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused(
                *qkv, gate, gate, state, torch.tensor([0, count], dtype=torch.int32), dim**-0.5
            )
    torch.testing.assert_close(actual, output)


@pytest.mark.parametrize("shape", [(2, 128), (1, 3, 4, 0)])
def test_head_first_l2norm_rejects_invalid_shape(shape):
    from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd_head_first

    with pytest.raises(ValueError, match="head-first L2Norm requires"):
        l2norm_fwd_head_first(torch.empty(shape))


@pytest.mark.parametrize("compact", [False, True])
def test_base_prefill_preserves_compacted_states_and_strided_inputs(compact):
    tokens, nk, nv, dim = 7, 2, 4, 16
    qkv = [torch.randn(1, h, tokens + 2, dim).transpose(1, 2)[:, 1:-1] for h in (nk, nk, nv)]
    normalized = [torch.randn(1, nk, tokens, dim) for _ in range(2)]
    state = torch.randn(2, nv, dim, dim)
    keep = torch.tensor([1]) if compact else None
    meta = SimpleNamespace(
        cu_seqlens_host=(0, 0, tokens),
        cu_seqlens_kern=(0, tokens),
        chunk_indices_chunk64_host=((0, 0),),
        keep_meta=keep,
    )
    gate = torch.randn(1, tokens, nv)
    output = torch.randn(1, tokens, nv, dim)
    final_kernel_state = torch.randn(1 if compact else 2, nv, dim, dim)

    def fused(q, k, v, *args, **kwargs):
        assert q is normalized[0] and k is normalized[1]
        torch.testing.assert_close(v, qkv[2].transpose(1, 2))
        expected_initial = state[keep] if compact else state
        torch.testing.assert_close(kwargs["initial_state"], expected_initial.transpose(-1, -2))
        assert kwargs["cu_seqlens"] == ((0, tokens) if compact else (0, 0, tokens))
        assert not kwargs["state_v_first"]
        assert not kwargs["use_qk_l2norm_in_kernel"]
        return output, final_kernel_state

    with patch("vllm_ascend.ops.triton.fla.l2norm.l2norm_fwd_head_first", side_effect=normalized) as normalization:
        actual, final_state = BaseDeviceAdaptor.fla_gdn_prefill(*qkv, gate, gate, state, dim**-0.5, meta, fused)
    assert normalization.call_args_list[0].args[0] is qkv[0]
    assert normalization.call_args_list[1].args[0] is qkv[1]
    assert actual is output
    expected_state = final_kernel_state.transpose(-1, -2)
    if compact:
        expected_state = torch.cat((state[:1], expected_state))
    torch.testing.assert_close(final_state, expected_state)


def test_head_first_l2norm_wide_heads_keep_upstream_fallback():
    from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd_head_first

    x = torch.randn(1, 3, 2, 1025)
    normalized = torch.randn_like(x)
    with patch("vllm.third_party.flash_linear_attention.ops.l2norm.l2norm_fwd", return_value=normalized) as norm:
        output = l2norm_fwd_head_first(x, eps=1e-5)
    assert norm.call_args.args[0] is x
    assert norm.call_args.kwargs["eps"] == 1e-5
    torch.testing.assert_close(output, normalized.transpose(1, 2))
    assert output.is_contiguous()
