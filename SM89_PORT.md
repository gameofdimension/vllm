# DeepSeek-V4-Flash on L20 (Ada, sm_89) — port notes

Branch **`v0.23.0-sm89`** (off tag `v0.23.0`): runs DeepSeek-V4-Flash-FP8
correctly on 8× NVIDIA L20 (Ada, sm_89, 46 GB). Verified output: `7×8→56`,
`"capital of France"→Paris`, `15×23→345`, `"Translate to French: hello
world"→bonjour le monde`. Server stable, no NaN.

> Sibling to `v0.23.0-sm120` (RTX PRO / sm_120). The two share the same
> approach (replace Hopper/Blackwell-only ops with Triton/PyTorch) and the
> sm89 branch builds on the sm120 port; the sm89-specific commits handle
> cutedsl ops and an fp8-checkpoint correctness bug that sm_120 never hit.

## Why a separate branch
V4's nvidia path targets Hopper/Blackwell: DeepGEMM (FP8 GEMM), FlashMLA-C,
FlashInfer TRTLLM-gen sparse MLA decode, cutedsl indexer/compressor, ue8m0
scales — all sm_90+ (some Blackwell-only). L20/Ada (sm_89) has none of these,
so every one must be routed to a portable fallback. All fallbacks are gated
on `is_deep_gemm_supported()` (False on sm_89) or `cap.major >= 9`, so
SM90/SM100 behavior is untouched.

## Commits on this branch (logical order)
1. **Port DeepSeek-V4 to non-DeepGEMM arches** — the sm120 port: Triton MLA
   decode, bf16 o_proj, PyTorch MLA prefill + indexer scorers.
2. **Route V4 cutedsl ops to Triton on sm_89** — `has_cutedsl()` arch-gate +
   compressor arch-gate (cutedsl emits sm_90+ PTX).
3. **Fix wo_a FLT_MAX placeholder scale** — fp8-checkpoint correctness: wo_a
   is stored BF16, gets a FLT_MAX scale on re-quant → o_proj dequant overflow
   → all-NaN. Use the weight's true values when the scale is corrupt.

## Requirements / setup (NOT in this repo — host side)
- **Driver ≥ 580 (CUDA 13.2)** matching the CUDA-13.2 toolchain (fixes
  Triton + tilelang JIT). Under driver 580 the system ptxas works; no
  `TRITON_PTXAS_PATH` override needed.
- **Checkpoint**: `/warehouse/DeepSeek-V4-Flash-FP8/config.json` must have
  top-level `"expert_dtype": "fp8"` (the sgl FP8 repack omits it; default
  `"fp4"` mis-loads the MoE).
- TP=8 mandatory (256 experts → TP must be a power of 2; TP=4 = 71 GB/card).
- GPUs are shared — verify all 8 free before launch (`nvidia-smi`); kill+relaunch
  races CUDA memory release.

## Serve
```bash
vllm serve /warehouse/DeepSeek-V4-Flash-FP8 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.9 --max-model-len 256000 \
  --enforce-eager --trust-remote-code
# (no --attention-backend flag: defaults to FlashMLA, which uses the Triton
#  decode on sm_89 via the port. --enforce-eager for fast iteration; cudagraph
#  gives little here since the decode runs in an eager-break segment.)
```
Verified at this config: ~519k-token KV cache, 2.03× concurrency at 256k.

## Caveats
- **Decode is slow on Ada** — the Triton MLA decode is the bottleneck (~89%
  of GPU time even on sm_120). Correctness is the goal of this port, not perf.
- Debugged against the **FP8** checkpoint only (the fp4 checkpoint is not on
  this host). The wo_a fix (commit 3) is fp8-specific.
- Full debug history / recipe: see the operator's `/root/kv-calc/HANDOFF.md` §13–§15.
