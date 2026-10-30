# Experimental DeepGEMM-Ascend MegaMoE integration

This isolated checkout adds an opt-in routed-expert backend for original
DeepSeek V4.1 FP4 / UE8M0 checkpoints on Ascend950. Default execution is unchanged.
The hardware-tested baseline revision is `8fde6d4e335fe300e2b949aab0e31659cce0811d`.
The integration is ported to `a4ff214c6a5991468b1baaf69c35c2c8a0f62339` for
publication. Hardware measurements below describe the original baseline;
the port needs a new hardware validation run. The default weight preparation
now delegates to the updated parent implementation.
DeepGEMM-Ascend is pinned to `3a35326271b069b8b5bf1a754435acd6cc91575b`.

## Design and restrictions

Set `additional_config.enable_deepgemm_mega_moe=true` together with
`enable_fused_mc2=2`. The optional package is imported only by the worker.
The existing `enable_fused_mc2` gate selects the fused communication path;
the new flag chooses DeepGEMM instead of CANN. CANN's shape whitelist is
not applied to DeepGEMM. Missing dependencies and unsupported configurations
fail early rather than silently running another backend.

The first version requires eager execution, TP=PP=PCP=DCP=1, EP enabled,
DP between 2 and 64 with equally partitioned experts, and fixed linear expert
placement. LoRA, speculation, EPLB, HCCL teardown/rebind, weight reload and
sleep mode are unsupported. Hardware validation here uses EP4 only.
Shared experts remain on the existing vLLM shared-expert path.

Checkpoint bytes are reinterpreted without numeric casting: FP4 as int8,
pairs of UE8M0 scales as int16. DeepGEMM interleaves gate/up blocks and
adjusts the FP4 scale exponents. The source expert tensors are then released.
A communicator-owned symmetric buffer is allocated once and reused across
sequential layers. Runtime input BF16 is dynamically quantized to MXFP8.
Routing indices and probabilities are copied into the symmetric allocation;
MegaMoE performs dispatch, both GEMMs, clamped SwiGLU and combine.
The integration explicitly requests `apply_router_weight_on_output=True`: it
quantizes the unweighted activation, runs GMM2, and multiplies routing weights
in the FP32 combine reduction, matching the existing MoE ordering. The upstream
DeepGEMM default multiplies weights before activation quantization, which is
mathematically equivalent without quantization but has different rounding.
The integration also sets `use_cann_mxfp_quant=True`: SwiGLU is rounded to
BF16 before quantization, scale is `2**(floor(log2(amax))-8)` per 32 values,
and normalized values are clipped to [-448, 448]. This matches the measured
CANN activation quantizer; upstream uses an FP32 activation and a different
scale selection. The isolated dependency patch adds both optional API flags
without changing their defaults.
An unpatched dependency is rejected when constructing the worker backend. Empty ranks
still participate in the collective. No device scalar is read in this hot path.

## Reproduction

The container's experiment harness is `/vllm-workspace/deepgemm-results`.
`env.sh` sets process-local paths to the isolated source, dependencies and JIT
cache. Pass `--quantization deepseek_v4_fp8` explicitly: this vLLM revision
does not auto-select it for `deepseek_v41`, and generic FP8 allocates incorrect
expert tensors. It preserves the CANN Python paths already present in the container.
`run-server.sh deepgemm` starts port 18901 with DP4/TP1, eager mode, context
4096, batch-token limit 2048, sequence limit 8, and CPU Engram offload.
The server uses one OpenMP thread, disables CPU binding and image inputs, and
unsets `LD_PRELOAD`. Earlier launches with other settings hit native crashes in
both backends; the tested combination avoids them, but the exact crash cause
has not been isolated.
`run-server.sh baseline` uses the same settings with fused MC2 disabled.
`run-server.sh baseline-mc2` additionally sets `enable_prefill_mc2=true` to
use the existing non-fused MC2 dispatch/GMM/combine path for these prompts.
This is the numerical reference with the same input quantization semantics.
This baseline needs `HCCL_BUFFSIZE=2048` for the expanded MC2 capacity;
the default baseline and DeepGEMM use 1024. The first prefill-MC2 launch
failed with a documented 4093 MB minimum window requirement.
The MC2 reference also uses memory utilization 0.93 rather than 0.90,
because the larger communication allocation otherwise leaves no KV cache.
These configuration differences are retained in the benchmark report.
The default unfused baseline switches to AllGather above its decode capacity
(8 tokens in this experiment). Its fused init-routing quantizer is byte-identical
to `npu_dynamic_mx_quant(scale_alg=1)` in the standalone comparison, while
MC2 and this DeepGEMM integration use `scale_alg=0`. Thus changing the
communication path also changes quantization numerics. Do not claim parity
with default AllGather prefill; its output/logprob differences are retained.
Do not run both simultaneously on the same four devices.

`examples/deepgemm/validate_real_weights.py` reads layer 0 expert weights from the actual checkpoint.
It compares MegaMoE with independent FP4 unpacking, BF16 matmuls, clamped SwiGLU,
MXFP8 quantization and FP32 rank reduction for 1, 7 and 32 tokens per rank, then
measures repeated calls for 1, 7, 32, 64 and 256 tokens. Its timer includes Python
launch overhead and warm caches; it is not a cold-L2 profiler measurement.
`probe_model.py` records eight deterministic completions and fixed-prompt
log-probabilities, then three streaming runs at concurrency 1, 4 and
16 with 64 generated tokens. The first run at each concurrency is treated as
warmup because it can trigger compilation; report the mean of the following
two runs and retain all raw measurements. A unique prefix on each performance request
prevents prefix-cache hits; both backends receive the same request sequence. The prompt and request data are saved with results.

The existing CANN MegaMoE whitelist rejects this model's first-projection
width 4608. The unfused baseline must not be labeled CANN MegaMoE performance.

## Dependency compatibility changes

The isolated DeepGEMM source lazily imports the unrelated TileLang HC kernel,
uses the only available HCCL network layer on single-layer topologies (still
requiring UB_MEM links to every peer), includes CANN debug intrinsics explicitly,
and uses `type_traits` instead of `utility` for device compiler compatibility.
Its extension is built with the CANN GCC14 toolchain and isolated elfutils headers.
The V4.1 attention path also requires the backed-up `cannbotdsl` and
`cannbot_arena_net_ops` wheels in the isolated dependency directory.
The DeepGEMM process-local `CC` selects GCC14 for Triton's host extension build
too; the baseline uses its normal system C++ runtime and Clang compiler.
The exact dependency diff is included at
`examples/deepgemm/deepgemm_ascend_compat.patch`. Apply it to the pinned
DeepGEMM revision before building. In addition to the compatibility changes,
it adds optional post-GMM2 routing weights: a reserved UB region stages one
32-byte DMA slot per route, and the combine vector kernel multiplies each
BF16 expert result in FP32 before summing. Its original default path is also
covered by the four-rank real-weight regression.
These patches and environment settings are required for this container and
must be reviewed separately before general deployment.

## Release note draft

Added an experimental, opt-in DeepGEMM-Ascend routed MegaMoE backend for
DeepSeek V4.1 FP4 checkpoints on Ascend950, with eager expert-parallel execution.
A numbered release-note fragment should be added when a PR number is assigned.

The reusable real-checkpoint validation script is also included in this checkout:

```bash
source /vllm-workspace/deepgemm-results/env.sh
python examples/deepgemm/validate_real_weights.py \
    --model /models/DeepSeek-V4.1-Flash-1017 \
    --output-dir /vllm-workspace/deepgemm-results/kernel-final \
    --num-processes 4
```

Run this only when the four devices are free. It also checks an empty rank
participating alongside three active ranks. The synthetic hidden states and
routes exercise actual checkpoint expert weights; this is not a model-quality
benchmark. Full-model checks are recorded separately by `probe_model.py`.

## Publication validation

The port to `a4ff214c6a5991468b1baaf69c35c2c8a0f62339` passed 183 CPU
unit tests using the branch-pinned vLLM revision
`544f182d9f0c9bfe8bd7645525951d5e3692a542`:

```bash
python -m pytest -q tests/ut/ops/test_deepgemm_mega_moe.py \
    tests/ut/ops/test_moe_comm_method.py tests/ut/test_ascend_config.py
```

The tests include default weight-preparation delegation and converted-weight
release/reload guards added for this port. No NPU run was performed on the
publication revision.

Ruff checks and formatting, codespell, typos, and the reverse-application check
of the dependency patch passed. The full `bash format.sh ci` invocation could
not complete because its Node.js environment download stalled and was stopped.
This is an incomplete full-repository lint result, not a passing result.
