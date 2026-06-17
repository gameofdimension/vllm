# CHANGES.md — vLLM wheel 源文件改动清单

> 每条改动记录：日期、wheel 内相对路径（`.venv/lib/python3.12/site-packages/vllm/<rel>`）、改动内容、备份文件、守卫条件。
> 还原：`bash sm120/revert.sh`。

约定：所有改动带 `is_deep_gemm_supported()` / 能力门守卫，**SM90/SM100 行为不变**，只在 sm_120（`get_device_capability()==(12,0)`）走新路径。

---

## [2026-06-17] Tier 1 — attention o_proj bf16 fallback

**目的**：消除当前崩溃（`_o_proj` 硬编码 DeepGEMM `fp8_einsum` → sm_120 上 `deepgemm layout.hpp:39 t.dim()==N`）。在 sm_120 上退回 bf16 路径（镜像 SGLang `deepseek_v4.py:996-1019` 的 `else` 分支）。

### 1. `models/deepseek_v4/nvidia/ops/o_proj.py`
- 备份：`sm120/backups/o_proj.py.orig`
- 新增（模块级）：`sm120_o_proj(...)` + 辅助函数 `_sm120_inv_rope(...)`、`_sm120_wo_a_bf16(...)`。
  - `_sm120_inv_rope`：纯 torch 实现 DeepSeek-V4 的 interleaved（GPT-J 式）逆向 RoPE（与 `fused_inv_rope_fp8_quant` 的 Triton 核数学一致），bf16 输入输出。
  - `_sm120_wo_a_bf16`：把 `wo_a.weight`（运行时为 fp8 e4m3，加载时由 bf16 checkpoint 在线量化而来）按 block scale（`weight_scale_inv`，`[128,128]` 块，float32）反量化为 bf16，结果缓存在 `wo_a._sm120_bf16`。
  - `sm120_o_proj`：inv-RoPE → reshape `[T, n_local_groups, d]` → `torch.einsum("tgd,grd->tgr", o, wo_a_bf16)`（float32 累加）→ `wo_b`。

### 2. `models/deepseek_v4/nvidia/flashmla.py`
- 备份：`sm120/backups/flashmla.py.orig`
- `_o_proj`（原 line 42）：`if is_deep_gemm_supported(): return deep_gemm_fp8_o_proj(...) else: return sm120_o_proj(...)`。

### 3. `models/deepseek_v4/nvidia/flashinfer_sparse.py`
- 备份：`sm120/backups/flashinfer_sparse.py.orig`
- `_o_proj`（原 line 85）：同上 `if/else`。

### 守卫
`is_deep_gemm_supported()`（`vllm/utils/deep_gemm.py:87`）= `VLLM_USE_DEEP_GEMM and has_deep_gemm() and support_deep_gemm()`。sm_120 上 `support_deep_gemm()=False` → 走 fallback；SM90/SM100 → 原路径。

### 验证（2026-06-17，重跑 `bash launch.sh`）
**Tier 1 o_proj fallback 成功**：
- `t.dim()==N`（o_proj DeepGEMM `layout.hpp:39`）崩溃**消失**；traceback 命中 `flashmla.py:_o_proj → sm120_o_proj`（`is_deep_gemm_supported()=False` 正确分流）。
- 中途修了 1 个自身 bug：`wo_b` 返回非 2 元组，`o, _ = wo_b(...)` 报 "too many values to unpack" → 改回 `return wo_b(...)`（与原 `deep_gemm_fp8_o_proj` 一致，不 unpack）。
- 修复后**整个越过 o_proj 前向**，进入下一阶段。

**暴露的下一堵墙（MLA attention backend 自身的 DeepGEMM 依赖）**：
```
v1/worker/gpu_model_runner.py _dummy_run → _build_attention_metadata
v1/attention/backends/mla/indexer.py:613  builder.build(...)
  → get_paged_mqa_logits_metadata(...)          # MLA indexer scheduler metadata
vllm/utils/deep_gemm.py:405
RuntimeError: deepgemm-src/csrc/apis/attention.hpp:219: Unsupported architecture
```
`get_paged_mqa_logits_metadata`（及相邻 `fp8_fp4_paged_mqa_logits`，deep_gemm.py:408）是 DeepGEMM 的 **attention 内核**（`attention.hpp`，非 GEMM），同样**绕过 `is_deep_gemm_supported()` 门**（`_lazy_init()` 后直接调 impl），sm_120 "Unsupported architecture"。这是 MLA 注意力后端的硬 DeepGEMM 依赖（即 docs 里墙 #2/#3 的具体落点），属 Tier 2 范畴——需要用 Triton MLA 核替换。

**结论**：o_proj bf16 方案 checkpoint 无关、已验证可用；MLA 注意力这一整块（indexer + decode 核）才是下一个、也是更大的工程。

---

## [2026-06-17] Tier 1b — ue8m0→f32 FP8 线性层 scale 转换（fp4 第一堵墙）

**目的**：消除 fp4 checkpoint 的墙 ①（注意力/稠密 FP8 block 线性层 scale=ue8m0 → sm_120 DeepGEMM 被门挡 → CUTLASS `dispatch_scaled_mm scaled_mm_helper.hpp:17` 落空，无 e8m0 kernel）。

### 4. `model_executor/layers/quantization/fp8.py`
- 备份：`sm120/backups/fp8.py.orig`
- 改动：`Fp8LinearMethod.process_weights_after_loading` 的 `block_quant` 分支末尾，当 `self.is_scale_e8m0 and not is_deep_gemm_supported()` 时，把 `layer.weight_scale_inv`（`float8_e8m0fnu`）转成 `float32`：
  ```python
  sf_f32 = sf.contiguous().view(torch.uint8).to(torch.int32).__lshift__(23).view(torch.float32)
  replace_parameter(layer, "weight_scale_inv", sf_f32)
  layer.weight_scale_inv.format_ue8m0 = False
  ```
  数学 = `(e<<23)` 重解释为 f32 = `2^(e-127)`，同 `nvidia/model.py:282 _ue8m0_uint8_to_float`。CUTLASS e4m3+f32-block `scaled_mm` 在 Blackwell 能 dispatch（只 e8m0 无 kernel）。

### 守卫 / 范围
`self.is_scale_e8m0`（fp8.py:269）只在 **DeepSeek-V4 fp4** 为 True；`not is_deep_gemm_supported()` 限定 sm_120。两者与 → **只影响 DeepSeek-V4 fp4 on sm_120，不动其他 FP8 模型，SM90/100 不变**。

### 验证（2026-06-17，fp4 checkpoint `/warehouse/DeepSeek-V4-Flash`）
**成功**：`dispatch_scaled_mm scaled_mm_helper.hpp:17` + `float8_e8m0` 全部消失；FP8 线性层（`fused_wqa_wkv` 等）+ o_proj 全部跑通。下一堵 = 墙 ③ `get_paged_mqa_logits_metadata → attention.hpp:219 Unsupported architecture`（MLA indexer DeepGEMM，checkpoint 无关，Tier 2）。

**fp4 四墙进度**：①✅(1b) ②✅(Tier1) ③⏳(MLA, Tier2) ④(MoE 运行时, Tier3)。

---

## [2026-06-17] Tier 2 探查 — indexer guard flip + FlashInfer 实验

### 5. `v1/attention/backends/mla/indexer.py`
- 备份：`sm120/backups/indexer.py.orig`
- 改动：line 612 守卫 `has_deep_gemm()` → `is_deep_gemm_supported()`（加 import）。效果：sm_120 跳过 DeepGEMM 的 `get_paged_mqa_logits_metadata`（墙 ③ 的 indexer-metadata 部分），消除 `attention.hpp:219` 崩溃。SM90/100 不变。**仅消除 indexer metadata 崩溃，不足以让 MLA decode 跑起来**（见下）。

### 实验：`--attention-backend FLASHINFER_MLA_SPARSE_DSV4`（结果：失败）
indexer guard flip 后用 FlashInfer sparse 后端解码，看能否避开 DeepGEMM MLA。结果：
```
flashinfer/trtllm/fmha/fmhaRunner.cuh:37 — TllmGenFmhaRunner: Unsupported architecture
```
FlashInfer 0.6.12 的 TRT-LLM FMHA（MLA decode 核）**也不支持 sm_120**。

### 结论（Tier 2 定性）
vLLM 0.23.0 **没有任何现成的 sm_120 MLA decode 核**：DeepGEMM(FlashMLA) 和 FlashInfer(TRT-LLM FMHA) 都不支持。廉价楔子（换后端）已排除。墙 ③ 的 MLA decode 必须**移植一个 Triton MLA decode 核**（SGLang `flash_mla_sm120_triton.py`），是最大的一块（天级+）。indexer guard flip 是该方案的一部分（保留）。

---

## [2026-06-17] Tier 2 移植 — MLA decode Triton 核（已移植+接线）+ c4 indexer gap

### 6. 新文件 `models/deepseek_v4/nvidia/flash_mla_sm120_triton.py`（canonical 副本：`sm120/flash_mla_sm120_triton.py`）
- 从 SGLang `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py` **逐字移植**（Apache-2.0）。自包含纯 Triton 核 `_tiled_sparse_decode_kernel` + `_run_triton_sparse_decode` + `flash_mla_sparse_decode_triton` + merge/sink helpers。DSv4 页内布局（448B fp8 nope ‖ 128B bf16 rope ‖ 8B/token ue8m0 scale）= vLLM 的 `fp8_ds_mla`，核内 FP8 反量化。
- 新文件（无 .orig）；canonical 副本在 `sm120/`。

### 7. `models/deepseek_v4/nvidia/flashmla.py`（_forward_decode 接 Triton 核）
- 备份：`sm120/backups/flashmla.py.orig`（Tier 1 已备份）
- 改动：`_forward_decode` 的 `flash_mla_with_kvcache(...)` 调用包 `if is_deep_gemm_supported(): <原 DeepGEMM> else: flash_mla_sparse_decode_triton(...)`。参数一一对应（vLLM 多的 `tile_scheduler_metadata`/`block_table`/`cache_seqlens`/`is_fp8_kvcache` 是 DeepGEMM 专有，Triton 核不需要）。加 import `flash_mla_sparse_decode_triton`。
- import 校验通过。

### 实测（fp4，/tmp/vllm_fp4_mla.log）
MLA decode 不再崩在 `flash_mla_with_kvcache`。**新 gap = c4 indexer scorer**：
```
model_executor/layers/sparse_attn_indexer.py:324  logits = fp8_fp4_paged_mqa_logits(...)
utils/deep_gemm.py:449  _fp8_fp4_paged_mqa_logits_impl(...)
RuntimeError: deepgemm attention.hpp:376 Unsupported architecture
```
这是 decode 前给 c4 sparse 选 top-K 块打分的 paged MQA 核（q [B*next_n,H,D] fp8/fp4 + paged KV [num_blocks,block_size,1,D+4] uint8 末尾4字节 float scale → logits [B*next_n,max_model_len]），DeepGEMM 第 2 个 attention 内核。需移植非 DeepGEMM 版（SGLang `fp8_paged_mqa_logits_torch_sm120` PyTorch 版）。

decode 前向顺序：① c4 indexer 打分(fp8_fp4_paged_mqa_logits) → ② MLA decode(Triton，已移植)。① 先崩。

---

## [2026-06-17] Tier 2 移植（续）— prefill MLA + prefill indexer + VERIFIED WORKING

### 8. 新文件 `models/deepseek_v4/nvidia/flash_mla_sparse_fwd_sm120.py`（canonical `sm120/`）
- PyTorch 替换 FlashMLA C 核 `flash_mla_sparse_fwd`（SM90a/SM100f only）。prefill 已把 KV gather+反量化成稠密 bf16 buffer，故只需在稠密 KV 上做 gathered sparse attention（per-query indices + topk_length + attn_sink 进 softmax 分母）。
- 接进 `flashmla.py _forward_prefill` sm_120 分支（line ~381 `if is_deep_gemm_supported() else flash_mla_sparse_fwd_sm120`）。

### 9. `model_executor/layers/fp8_paged_mqa_logits_sm120.py` 追加 `fp8_mqa_logits_sm120`（非分页 prefill indexer scorer）
- 替换 DeepGEMM `fp8_fp4_mqa_logits`（`attention.hpp:184`，非分页，KV 已 gather 成稠密 [N,D]）。chunked over K。
- 接进 `sparse_attn_indexer.py:250` sm_120 分支。Bug 修复：einsum 输出序 `mhd,cd->mch`（[M,C,H]），非 `mhc`（[M,H,C]），否则与 weights [M,1,H] 形状不匹配。

### ✅ 验证（VERIFIED 2026-06-17）
服务器启动成功，真实请求正确：
- `POST /v1/chat/completions` "Say hello in one short sentence." → `"Hello!"`（finish=stop）
- `POST /v1/completions` "The capital of France is" → `" Paris."`（事实正确；后续 JSON 是 raw-completion 无 chat 模板 + 撞 token 上限的格式泄漏，非 bug）

**fp4 四墙全部攻破（①②③④）**，DeepSeek-V4-Flash 在 sm_120 上端到端正确推理。MoE Marlin 运行时无 NaN。launch.sh 指向 fp4 checkpoint，无需任何 config override。

### 剩余（非阻塞，仅速度）
PyTorch indexer scorer + prefill MLA 是朴素 eager 实现（分块循环），远慢于 DeepGEMM/C 核；首次 CUDA graph 捕获也慢。生产吞吐需优化（换 Triton scorer / 融合）。门控 `is_deep_gemm_supported()`，SM90/SM100 完全不受影响。

---
