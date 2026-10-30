"""Four-rank real-checkpoint MegaMoE check against independent BF16 matmuls."""

import argparse
import contextlib
import json
import time
from pathlib import Path

import deep_gemm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from deep_gemm.testing.envs import init_dist
from safetensors import safe_open

HIDDEN, INTERMEDIATE, EXPERTS, TOPK = 5120, 2304, 384, 6


def unpack(weight, scale):
    packed = weight.view(torch.uint8).cpu()
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device="cpu"
    )
    sf = torch.exp2(scale.view(torch.uint8).cpu().float() - 127).repeat_interleave(32, dim=-1)
    return (lut[nibbles] * sf).bfloat16().npu()


def quant_dequant(x):
    # Independent MXFP8 quantizer, one UE8M0 exponent per 32 channels.
    blocks = x.bfloat16().float().cpu().reshape(x.shape[0], -1, 32)
    scale = torch.exp2(torch.floor(torch.log2(blocks.abs().amax(-1).clamp_min(2.0**-119))) - 8)
    return (
        ((blocks / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale[..., None])
        .reshape(x.shape)
        .bfloat16()
        .npu()
    )


def worker(rank, world, model_path, output_path):
    model = Path(model_path)
    output_dir = Path(output_path)
    _, _, group = init_dist(rank, world)
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    local = EXPERTS // world
    with contextlib.ExitStack() as stack:
        files = {}

        def get(name):
            shard = index[name]
            if shard not in files:
                files[shard] = stack.enter_context(safe_open(str(model / shard), framework="pt", device="cpu"))
            return files[shard].get_tensor(name)

        l1, s1, l2, s2 = [], [], [], []
        for e in range(rank * local, (rank + 1) * local):
            prefix = f"layers.0.ffn.experts.{e}."
            l1.append(torch.cat([get(prefix + "w1.weight"), get(prefix + "w3.weight")], dim=0))
            s1.append(
                torch.cat(
                    [get(prefix + "w1.scale").view(torch.uint8), get(prefix + "w3.scale").view(torch.uint8)], dim=0
                )
            )
            l2.append(get(prefix + "w2.weight"))
            s2.append(get(prefix + "w2.scale").view(torch.uint8))
    a = (torch.stack(l1).npu().view(torch.int8), torch.stack(s1).npu().view(torch.int16))
    b = (torch.stack(l2).npu().view(torch.int8), torch.stack(s2).npu().view(torch.int16))
    aw, bw = deep_gemm.transform_weights_for_mega_moe(a, b)
    buffer = deep_gemm.SymmBuffer(group, EXPERTS, 256, TOPK, HIDDEN, INTERMEDIATE)
    results = []
    for tokens in (1, 7, 32, 64, 256):
        torch.manual_seed(900 + rank + tokens)
        x = torch.randn((tokens, HIDDEN), device="npu", dtype=torch.bfloat16) * 0.3
        xq, xs = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        ids = ((torch.arange(tokens * TOPK, device="cpu").reshape(tokens, TOPK) * 17 + rank * 97) % EXPERTS).npu()
        weights = torch.softmax(torch.randn((tokens, TOPK), device="npu", dtype=torch.float32), -1) * 1.5
        buffer.x[:tokens].copy_(xq)
        buffer.x_sf[:tokens].copy_(xs.contiguous().view(torch.int16).reshape(tokens, -1))
        buffer.topk_idx[:tokens].copy_(ids)
        buffer.topk_weights[:tokens].copy_(weights)
        y = torch.empty_like(x)

        def run(y=y):
            deep_gemm.fp8_fp4_mega_moe(
                y, aw, bw, buffer, activation_clamp=10.0, apply_router_weight_on_output=True, use_cann_mxfp_quant=True
            )

        run()
        torch.npu.synchronize()
        if tokens == 1:
            singleton_inputs = (xq.clone(), xs.clone(), ids.clone(), weights.clone())
            singleton_output = y.clone()
        row = {"tokens_per_rank": tokens, "rank": rank}
        if tokens <= 32:
            # Decode exactly the quantized input passed to MegaMoE.
            xd = (
                (
                    xq.float().cpu().reshape(tokens, -1, 32)
                    * torch.exp2(xs.view(torch.uint8).cpu().float().reshape(tokens, -1, 1) - 127)
                )
                .reshape(tokens, HIDDEN)
                .bfloat16()
                .npu()
            )

            def gather(t):
                ts = [torch.empty_like(t) for _ in range(world)]
                dist.all_gather(ts, t, group=group)
                return torch.cat(ts)

            all_x, all_ids, all_weights = gather(xd), gather(ids).cpu(), gather(weights)
            reference = torch.zeros((world * tokens, HIDDEN), device="npu", dtype=torch.float32)
            for e in range(local):
                ti, ki = torch.where(all_ids == rank * local + e)
                if not ti.numel():
                    continue
                ti, ki = ti.npu(), ki.npu()
                w1 = unpack(a[0][e], a[1][e])
                w2 = unpack(b[0][e], b[1][e])
                z = torch.nn.functional.linear(all_x[ti], w1).float()
                gate, up = z.chunk(2, -1)
                gate = gate.clamp(max=10)
                up = up.clamp(-10, 10)
                act = gate / (1 + torch.exp(-gate)) * up
                val = torch.nn.functional.linear(quant_dequant(act), w2).float() * all_weights[ti, ki, None]
                reference.index_add_(0, ti, val)
            dist.all_reduce(reference, group=group)
            ref = reference[rank * tokens : (rank + 1) * tokens].bfloat16().float()
            actual = y.float()
            diff = (
                (actual - ref).square().sum() / (actual.square().sum() + ref.square().sum()).clamp_min(1e-20)
            ).item()
            cos = torch.nn.functional.cosine_similarity(actual.flatten(), ref.flatten(), dim=0).item()
            row.update(
                normalized_squared_error=diff,
                cosine_similarity=cos,
                max_abs_error=(actual - ref).abs().max().item(),
                finite=bool(torch.isfinite(actual).all().item()),
            )
            assert row["finite"] and diff < 5e-5, row
        for _ in range(5):
            run()
        torch.npu.synchronize()
        dist.barrier(group)
        start = time.perf_counter()
        for _ in range(30):
            run()
        torch.npu.synchronize()
        row["wall_us_per_call"] = (time.perf_counter() - start) * 1e6 / 30
        print(json.dumps(row), flush=True)
        results.append(row)
    # Exercise a genuinely empty rank alongside active ranks using the same
    # singleton inputs as above. Other tokens must not change a token's output.
    xq, xs, ids, weights = singleton_inputs
    buffer.x[:1].copy_(xq)
    buffer.x_sf[:1].copy_(xs.contiguous().view(torch.int16).reshape(1, -1))
    buffer.topk_idx[:1].copy_(ids)
    buffer.topk_weights[:1].copy_(weights)
    empty_output = torch.empty((0 if rank == 0 else 1, HIDDEN), dtype=torch.bfloat16, device="npu")
    deep_gemm.fp8_fp4_mega_moe(
        empty_output,
        aw,
        bw,
        buffer,
        activation_clamp=10.0,
        apply_router_weight_on_output=True,
        use_cann_mxfp_quant=True,
    )
    torch.npu.synchronize()
    if rank:
        torch.testing.assert_close(empty_output, singleton_output, rtol=0.01, atol=0.001)
    results.append({"rank": rank, "empty_rank_collective": "passed", "local_tokens": empty_output.shape[0]})
    (output_dir / f"kernel-rank{rank}.json").write_text(json.dumps(results, indent=2))
    dist.barrier(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-processes", type=int, default=4)
    args = parser.parse_args()
    config = json.loads((Path(args.model) / "config.json").read_text())["text_config"]
    assert (
        config["hidden_size"],
        config["moe_intermediate_size"],
        config["n_routed_experts"],
        config["num_experts_per_tok"],
    ) == (HIDDEN, INTERMEDIATE, EXPERTS, TOPK)
    assert EXPERTS % args.num_processes == 0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    mp.spawn(worker, args=(args.num_processes, args.model, args.output_dir), nprocs=args.num_processes, join=True)
