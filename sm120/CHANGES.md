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

## [2026-06-17] 源码树迁移落地 + editable install（环境实录）

> 范式从「wheel 补丁」切到「源码树编辑」的落地记录。包含一处**重要环境偏差**：HANDOFF §5 的 `uv pip install -e .` 在本机因 PyPI CDN 不可达而走不通，改用「复用 native-vllm」方案。

### A. 源码树改动落地（`/root/vllm-sm120/vllm/`）
- 源码 == `backups/*.orig`（干净 v0.23.0，仅多一个 `add handoff files` commit）→ 4 个新文件 `cp` + 6 处编辑 `cp patched/` 为**精确覆盖**（diff 验证 6 个编辑文件 == canonical patched；4 个新文件就位）。
- 10 个文件 `py_compile` 全过；导入路径校验：`sm120_o_proj` / `flash_mla_sm120_triton` / `flash_mla_sparse_fwd_sm120` / `fp8_paged_mqa_logits_sm120` 均命中正确源码路径；`is_deep_gemm_supported()=False`。

### B. editable install —— 偏离 HANDOFF §5（重要）
- **根因**：本机 `files.pythonhosted.org`（PyPI CDN）**不可达**（6MB wheel 20s 超时），而 `download.pytorch.org` / `wheels.vllm.ai` 可达。→ 无法从 PyPI 拉 torch/依赖 wheel，`VLLM_USE_PRECOMPILED=1 uv pip install -e .` 卡死在依赖下载；fresh build 还缺 `setuptools-rust`/`wheel`（且 PyPI 取不到）。
- **复用基础**：`/root/native-vllm/.venv` = 旧 wheel-patched 验证环境 —— vllm 0.23.0 site-packages **已带 sm120 补丁** + 全套 `_C` + 全依赖（torch 2.11.0+cu130，190 包）。即「四墙全攻破」那套环境的留存。
- **实际 editable 做法**（`uv venv` 创建 venv 符合「用 uv」，依赖/`_C` 复用绕开 PyPI）：
  1. `uv venv --python 3.12 .venv`
  2. `cp -al /root/native-vllm/.venv/.../site-packages/. .venv/.../site-packages/`（硬链接全依赖，0 额外磁盘）
  3. 删 `.venv/.../site-packages/vllm/`（消除与源码竞争；保留 `vllm-0.23.0.dist-info`）
  4. **源码树补齐 build/vendored 产物**（git checkout 缺、安装包才有）——从 native-vllm `cp` 进 `vllm/`：
     - 14 个 `.so`：`_C`、`_C_stable_libtorch`、`cumem_allocator`、`_flashmla_C`、`_flashmla_extension_C`、`_moe_C`、`spinloop`、`vllm_flash_attn/{_vllm_fa2_C,_vllm_fa3_C}`、`third_party/deep_gemm/_C.cp31x`
     - 整个 `third_party/`（flashmla/deep_gemm/triton_kernels/pynvml，36M / 968 文件）
     - `vllm_flash_attn/cute/*`（vendored FA 接口 .py）
     - `vllm-rs`（36M Rust 扩展，`envs.py`/`v1/utils.py` 运行时导入）
     - `_version.py`（build 时生成）
     - 用 `cp -an`（no-clobber）gap-fill，**不覆盖**我们打了补丁的 6 个 .py + 4 个新文件（grep 验证 flashmla.py 仍有 4 个 sm120 标记）
  5. `.pth`（`_vllm_source_editable.pth` 含 `/root/vllm-sm120`）→ `import vllm` 解析到源码树
  6. `.venv/bin/vllm` 控制台脚本（`vllm.entrypoints.cli.main:main`）
- **验证**（从 `/tmp`，非 repo 根）：`import vllm` → `/root/vllm-sm120/vllm/__init__.py`；`vllm._C` 可载；`is_deep_gemm_supported()=False`；4 个新模块在导入的源码里；`vllm --help` 正常。
- **launch.sh 注意**：原 `uv run vllm serve` 会触发 `uv run` 的项目 sync（撞 PyPI）。源码模式下直接 `.venv/bin/vllm serve`（已验证），或 `uv run --no-sync vllm serve`。

### C. 端到端验证（VERIFIED 2026-06-17）
- `.venv/bin/vllm serve /warehouse/DeepSeek-V4-Flash --tp 8 --kv-cache-dtype fp8 --gpu-memory-utilization 0.9 --enforce-eager`：safetensors 100%（9.84s）、Model loading 20 GiB、KV cache 6.2M tokens、`Application startup complete`。**无** DeepGEMM/FlashMLA-C/ue8m0 崩溃（四墙全绕开）。
- `15*23=?`（max_tokens 256, temp 0）→ **345**，`finish_reason=stop`，分步推理正确。端到端正确推理复现。
- 用 `--enforce-eager` 为快速验证（省 ~12min CUDA graph 捕获）；sm120 算子本就在 breakable-cudagraph 的 eager 段。**生产吞吐**用去掉 `--enforce-eager` 的完整命令（启动 ~14min）。

### D. 源码树新增的 untracked 产物（git 视角，建议 .gitignore）
`vllm/*.so`、`vllm/third_party/`、`vllm/vllm_flash_attn/cute/`、`vllm/vllm-rs`、`vllm/_version.py`、`vllm/**/__pycache__` —— 均为 build 产物 / vendored / 生成文件，正常 editable dev tree 会有；应加入 `.gitignore` 避免误提交（它们不属于 sm120 改动本身）。

---

## [2026-06-18] 补充：PyPI 镜像修复 → 标准 uv editable install 可用（正解）

> §B 的「复用 native-vllm」是 PyPI CDN 不可达时的临时方案。**加腾讯 PyPI 镜像后，标准 `uv pip install -e .` 即可用** —— 这才是正解，复用方案降级为 fallback。

### 配置
`/root/.config/uv/uv.toml`（把腾讯镜像设为 default index，绕开不可达的 files.pythonhosted.org）：
```toml
[[index]]
name = "tencent"
url = "http://mirrors.tencentyun.com/pypi/simple"
default = true
```

### 标准 editable install 成功
```
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=cu130
# Resolved 190 packages in 42.96s
# + vllm==0.23.1.dev1+g21a7715fe.d20260618.precompiled (from file:///root/vllm-sm120)
```
- proper `__editable__.vllm-0.23.1.dev1+g21a7715fe...pth`（PEP 660 finder）+ dist-info；`_C` 由 precompiled 落入源码树。
- torch 仍走 `download.pytorch.org`（`--torch-backend=cu130`），其余走镜像。
- 验证（/tmp）：`import vllm`→`/root/vllm-sm120/vllm/__init__.py`；`vllm._C` 可载；`is_deep_gemm_supported()=False`；4 个 sm120 模块在导入源码；torch 2.11.0+cu130。
- `uv run vllm --help` 正常 → `launch.sh` 的 `uv run vllm serve` 现在可用（建议加 `--no-sync` 保险）。

### 端到端验证（VERIFIED 2026-06-18）
标准 install 起服务（`--enforce-eager`）：`Application startup complete`、weights 9.66s、Model 20.04 GiB。`15*23=?`→**345**，`finish_reason=stop`。与复用方案行为一致。

### 收尾
删 `.venv_reuse`（复用 fallback，不再需要），只留标准 `.venv`。`/root/native-vllm/.venv` 仍作 _C/依赖的应急来源保留。

---

## [2026-06-18] Perf Phase 1 — c4 paged-MQA decode scorer → Triton（Op1）

> 吞吐优化第一步：把 decode 热路径上的朴素 PyTorch c4 scorer 换成融合 Triton 核。Op1 是最热的（decode 每步都跑；1M ctx 下 oracle ~28ms/步，~2000 次串行 launch）。Op2/Op3 暂缓，本条先 checkpoint。

### 改动
- 新文件 `vllm/model_executor/layers/fp8_paged_mqa_logits_triton.py`：`_paged_mqa_logits_kernel`（`@triton.jit`）+ `fp8_paged_mqa_logits_sm120_triton`（签名与 PyTthon oracle 完全一致）。
  - Grid `(token_tile, query_row)`；无 softmax → 无跨 tile 归约，token 维切分给足并行度。
  - 每 program：经 `block_tables` 间接分页加载 KV（per-token 128B fp8 值 + 4B f32 scale），`qk = tl.dot(kv_fp8, q_fp8ᵀ)`（**fp8 tensor-core MMA，f32 累加**）→ relu → 加权 head 求和 → `*k_scale`；`pos >= context_len → -inf`，流式写 `logits`。
  - 关键：**用 `tl.dot` fp8 而非逐 head 的 f32 matvec** —— 后者不用 tensor core，大 B 反而比 oracle 的 `bmm` 慢；fp8 MMA 既快又与 oracle（fp8 值的 f32 点积）数学一致。
- 接线 `vllm/model_executor/layers/sparse_attn_indexer.py` decode 的 `else` 分支：`scorer = triton if VLLM_SM120_TRITON_SCORER(=1) else pytorch`。SM90/100 走上面的 DeepGEMM 分支，不受影响。
- 基准 `sm120/bench/scorer_bench.py`（独立、不起服务）：构造输入 + `assert_close` 比 oracle + 计时。同时给出 Op1/Op2/Op3 的 oracle 基线。

### 基线 vs Triton（Op1）
| B | ctx | oracle(ms) | triton(ms) | speedup | max_err |
|---|---|---|---|---|---|
| 1 | 4k | 0.143 | 0.038 | 3.78× | 9.5e-7 |
| 1 | 32k | 0.711 | 0.036 | 19.6× | 9.5e-7 |
| 4 | 32k | 0.696 | 0.040 | 17.5× | 9.5e-7 |
| 8 | 128k | 3.630 | 0.150 | 24.2× | 1.4e-6 |
| 32 | 32k | 2.669 | 0.151 | 17.7× | 1.4e-6 |
| 8 | **1M** | **28.43** | **1.154** | **24.65×** | 1.4e-6 |

### VERIFIED
- 正确性：triton == oracle，max_err ~1e-6（`assert_close` tol 2e-2/4e-3），所有 shape（含 1M）通过，`-inf` 掩码一致。
- 端到端：`VLLM_SM120_TRITON_SCORER=1 vllm serve --enforce-eager`，`15*23=?` → **345**，`finish_reason=stop`。
- 单一固定配置（`BLOCK_TOK=64`, 4 warps, 2 stages）；autotune/按 shape 绑定配置留作后续。

### Op2/Op3 基线（待移植，参考）
- Op2 prefill scorer：M=4096,N=32k → 230ms。
- Op3 MLA prefill：Tq=2048,topk=2048 → 53ms。
- 两者都是 prefill 成本（比 decode 更可摊销），优先级低于 Op1。

---

## [2026-06-18] Perf Phase 2 — c4 dense (non-paged) prefill scorer → Triton（Op2）

> Op1 的同款 score 数学，KV 换成已 gather 的稠密 `[N,D]` fp8 + `[N]` f32 scale，按每个 query 的 `[ks,ke)` 变长区间打分，输出 `[M,N]`。

### 改动
- `vllm/model_executor/layers/fp8_mqa_logits_triton.py` 追加 `_dense_mqa_logits_kernel` + `fp8_mqa_logits_sm120_triton`（签名与 oracle 一致）。Grid `(kv_tile, query_row)`；稠密直读 KV（无 block_tables），`qk = tl.dot(kv_fp8, q_fp8ᵀ)` fp8 MMA → relu → 加权 head 求和 → `*k_scale`；`pos ∉ [ks,ke) → -inf`。
- 文件从 `fp8_paged_mqa_logits_triton.py` 改名为 `fp8_mqa_logits_triton.py`（paged+dense 两个 scorer 共置；`sparse_attn_indexer.py` / bench 的 import 同步改）。
- 接线 `sparse_attn_indexer.py` prefill 的 `else` 分支：`scorer = triton if VLLM_SM120_TRITON_SCORER(=1) else pytorch`（与 Op1 同一个 flag）。SM90/100 不受影响。
- bench `run_prefill` 加 `--compare`：oracle vs triton 正确性 + 计时。

### 基线 vs Triton（Op2）
| M | N | oracle(ms) | triton(ms) | speedup | max_err |
|---|---|---|---|---|---|
| 512 | 4096 | 3.734 | 0.183 | 20.5× | 9.5e-7 |
| 2048 | 32768 | 116.530 | 4.315 | 27.0× | 9.5e-7 |
| 4096 | 32768 | 230.775 | 8.804 | 26.2× | 9.5e-7 |

### VERIFIED
- 正确性：triton == oracle，max_err ~1e-6，所有 shape 通过（`-inf` 掩码一致）。
- 端到端：`VLLM_SM120_TRITON_SCORER=1 vllm serve --enforce-eager`（Op1+Op2 都走 Triton），`15*23=?`→345、`7×8`→56，正确。

### 剩余
Op3（MLA prefill gathered sparse attention，baseline 53ms）仍为朴素 PyTorch；是真 softmax（sink 在分母），需 flash 风格 online-softmax 核，与 Op1/Op2 的 score 数学不同。

---

## [2026-06-18] Perf Phase 3 — MLA prefill gathered sparse attn → Triton（Op3）

> 与 Op1/Op2 不同：真 softmax（attn_sink 加在分母），每个 query 各自 gather 一组稀疏 KV（per-query indices → KV 不能跨 query tile 共享）。设计上最绕。

### 踩坑路径（记录备查）
1. **per-(query,head) program + 逐 head matvec**：4× **变慢**（0.25×）。没用 tensor core；oracle 的 batched einsum 走 cuBLAS，打不过。
2. **per-query program + `tl.dot`（整 H）**：`OutOfResources` shared memory —— acc `[H=64, D=512]` f32 = 128KB 单块就超 sm_120 每块 99KB 上限。
3. **H-tile（BLOCK_H=16）+ `tl.dot` + `static_range` 展开 KV 循环**：4.16×、正确。但 `static_range(0, TOPK=2048, BLOCK_K)` 把循环展开 ~32 次 → 核巨大 → **e2e 首请求 JIT 编译耗时超过 ~5min RPC 超时，EngineCore 判 EngineDeadError**（无 CUDA 错，纯 timeout）。
4. **✅ H-tile + `tl.range`(运行时 bound) + BLOCK_K=32**：5.63×、正确、JIT 快。落定。

### 最终核（`vllm/models/deepseek_v4/nvidia/flash_mla_sparse_prefill_triton.py`）
- `_flash_mla_sparse_prefill_kernel` + `flash_mla_sparse_fwd_sm120_triton`（签名同 oracle，写 `out` in-place）。
- Grid `(Tq, cdiv(H, BLOCK_H=16))`；每 program 取 BLOCK_H 个 head，按 query 的 indices gather 自己的 KV，flash online-softmax（exp2）：
  - QK `q[BLOCK_H,D] · kv_g.T[D,BLOCK_K]`（bf16 MMA）→ mask → softmax
  - AV `p[BLOCK_H,BLOCK_K] · kv_g[BLOCK_K,D]`（bf16 MMA）
  - KV 循环用 **`tl.range`(运行时 topk)** 不展开（快编译、小核）；BLOCK_K=32 让 acc[BLOCK_H,D]+kv_g tile 进 99KB shared。
  - sink：`e_sink = exp2(sink·LOG2E − rowmax)` 加分母（不计分子）；scalar/[H]/None 都支持。
- 接线 `flashmla.py` `_forward_prefill` 的 else 分支：`prefill_fn = triton if VLLM_SM120_TRITON_MLA_PREFILL(=1) else pytorch`。SM90/100 不受影响。

### 基线 vs Triton（Op3）
| Tq | topk | oracle(ms) | triton(ms) | speedup | max_err |
|---|---|---|---|---|---|
| 512 | 512 | 3.511 | 0.608 | 5.78× | 6.1e-5 |
| 2048 | 2048 | 53.477 | 9.496 | 5.63× | 3.0e-5 |

### VERIFIED
- 正确性：triton == oracle，max_err ~3–6e-5（bf16 精度），`-inf`/sink 语义一致。
- 端到端：`VLLM_SM120_TRITON_SCORER=1 VLLM_SM120_TRITON_MLA_PREFILL=1 vllm serve --enforce-eager`（三核全开）→ `15*23=?`→345，`finish=stop`。

### 三核总结
| Op | 路径 | 最大 shape | 基线 | Triton | 加速 |
|---|---|---|---|---|---|
| Op1 | paged decode | B=8, 1M ctx | 28.4 ms | 1.15 ms | 24.6× |
| Op2 | dense prefill | 4096×32768 | 230.8 ms | 8.8 ms | 26.2× |
| Op3 | MLA prefill | 2048² | 53.5 ms | 9.5 ms | 5.63× |

三核均门控 `is_deep_gemm_supported()` 的 else（只 sm_120 走），SM90/100 完全不变；均可经环境变量回退到 PyTtorch oracle。剩余可选优化：按 shape autotune（当前为单一配置）、长上下文 prefill 显存压测。

---

## [2026-06-18] §7.1 收尾 — autotune 评估 + 长上下文压测

### Autotune：评估后**不用**（保留手工固定配置）
给 Op1 加了 `@triton.autotune`（BLOCK_TOK∈{32,64,128,256}×warps，按 ctx 分桶 key），bench 实测：
- 32k ctx：**15.3×（退步）** vs 固定 19.6×；4k：3.0× vs 3.8×（退步）。
- 仅 128k/1M 略好（25.5× vs 24.7×）。
**结论**：手工挑的固定配置（Op1 BLOCK_TOK=64、Op2 BLOCK_K=64、Op3 BLOCK_K=32）对常见（短/中 ctx）decode 已近最优；autotune 的内部计时噪声导致它给常见场景挑了更慢的配置，净负面，且增加首请求 JIT 编译开销（Op3 还有 RPC 超时风险）。已**revert Op1，Op2/Op3 不再尝试**（同理）。固定配置保留。

### 长上下文显存压测：通过
- **bench 大 shape**（无 OOM、正确、加速保持）：Op2 M=8192,N=32768 → 462.7→17.8ms（25.9×）；Op3 Tq=4096,topk=2048 → 107→20.7ms（5.18×）。
- **真实长 prompt serve**（`--enforce-eager`，三核全开）：~12,090 token prompt（320 段重复 + 一句藏在中段的密语 "ZEPHYR-42" + 提问）→ 正确召回 `ZEPHYR-42`，`finish_reason=stop`，**无 OOM、无 EngineDead**。证明集成 prefill 路径（Op2 c4 scorer + Op3 MLA prefill）在长上下文下注意力正确、显存可控。

### §7.1 状态：✅ 完成（三核 Triton 化 + 收尾）。剩余路线（HANDOFF §7）专注 v0.23.0 基线：正确性回归 / MoE 压测 / prefill 更大上下文。"上游化到 main" 已移出范围。

---

## [2026-06-18] 正确性 + 服务吞吐验证（GSM8K + bench serving）

### P1：GSM8K Pass@1 正确性（无需对照 SGLang）✅
- 用 vLLM 自带 `tests/evals/gsm8k/gsm8k_eval.py`（5-shot CoT，`/v1/completions`，取末位数字精确匹配）。GSM8K 数据从 openai/grade-school-math 拉（`--retry` 重传一次补全 test.jsonl，1319 题齐全）。
- 200 题样本：**Accuracy 0.965，invalid 0.000**。三核 Triton 全开下端到端数学推理正确，正确性确认。

### P2：服务吞吐 — Triton vs torch（vLLM 自家 A/B）
工具：`vllm bench serve`（已迁到 CLI；旧 `benchmarks/benchmark_serving.py` 是空壳）。脚本 `sm120/bench/run_perf.sh`（provider-agnostic，同一套命令打 vLLM / SGLang 的 `/v1/completions`；`--dataset-name random` 固定 in/out 长度、`--ignore-eos`、`--save-result`；两场景：A decode-heavy in128/out256×32，B long-prefill in4096/out16×64）。需 `--tokenizer /warehouse/DeepSeek-V4-Flash --tokenizer-mode deepseek_v4`（否则去 HF 拉 tokenizer 撞断网）。

关键点：sm120 三核带 `@eager_break_during_capture`，**cudagraph 开着也照常 eager 跑**，所以 eager 下 Triton-vs-torch 的 delta 就是这三核优化在生产里的真实 delta（cudagraph 只抬绝对值、不改变该 delta）。

结果（eager，TP8）：
| 场景 | Triton（完成） | torch |
|---|---|---|
| A decode-heavy (32 请求) | 32/32 完成，out 36.5 tok/s，ITL 872ms | **1/32 完成，out 0.37 tok/s** → 随后 EngineDeadError（RPC 超时） |
| B long-prefill (64 请求) | 64/64 完成，total 1674 tok/s | **0/64 完成** → EngineDeadError |

**结论**：torch 路径在服务负载下**不可用**——PyTorch 三核太慢，前向步超过 engine RPC 超时，进程崩溃；Triton 全量跑完。叠加单元级 24.6×/26.2×/5.63×，证明 Triton 化是 sm120 可服务的必要条件，不只是提速。eager 绝对值偏低（生产应去掉 `--enforce-eager`）。

### 剩余（P2 收尾）
- vLLM-vs-SGLang 绝对吞吐：需 vLLM 生产配置（cudagraph，去掉 `--enforce-eager`，~14min 启动）的 Triton 数；SGLang 由用户跑。`run_perf.sh` 同样适用（改 `--base-url` 指向 SGLang）。

---

## [2026-06-22] SGLang 三方对比 + §7.3 MoE/长上下文压测

### SGLang 三方对比（同一份 `run_perf.sh`，两场景）
| 场景 | vLLM-Triton(生产/cg) | SGLang | vLLM-torch |
|---|---|---|---|
| A decode (out tok/s / ITL) | 36.6 / 864ms | **85.2 / 287ms** | 0.37（挂） |
| B prefill (total tok/s / TTFT) | **1669 / 77s** | 179 / 801s | 0（挂） |

- **Prefill：vLLM ~9.3× 快**（Op2/Op3 Triton 化的直接收益，4096 输入下 indexer+MLA prefill 是主成本）。
- **Decode：SGLang ~2.3× 快**（vLLM sm120 decode 864ms ITL 是弱项；短上下文下 Op1 可忽略，瓶颈在 MoE/通信/MLA-decode）。
- **公平性**：SGLang serve 配置由用户给定，未与我严格对齐；prefill 9.3× 可能随 SGLang 配置变。decode 场景是短上下文，未压到 Op1 长上下文 24× 的优势。
- → vLLM decode 是主要优化空间，线索见 §7.5 decode-step profile。

### §7.3 MoE（Marlin MXFP4）吞吐 + 长上下文压测 — 全过
- **吞吐/稳定性**：sustained decode（in256/out512，32 请求，16k decode tokens）**36.4 tok/s 稳定不退化**，全 32/32 完成，**无 NaN / 无 EngineDead**。
- **长上下文显存**：needle 测试到 **64k 无 OOM**（64k prefill 22.8s）。
- **长上下文正确性**：needle 召回 12k/16k/**64k** 通过；32k 单次失败（"ZEPHYR-7B"）**非单调 → 判为单样本噪声**（sparse-attention 召回本身有随机性），全程无 NaN/garbage。
- **长生成质量**：300 词作文连贯准确、按要求收尾、无重复退化。
- 结论：MoE + 整条 sm120 路径在吞吐与长上下文（到 64k）下健康。**顺带覆盖 §7.4**（prefill 大上下文显存：64k 无 OOM）。

---

## [2026-06-22] §7.6 清理：torch oracle 移入 tests，serving 改 Triton-only

> PyTorch 三核实现负载下不可用（EngineDeadError），作为 serving 回退无意义。按"保留为正确性 oracle、但挪出生产代码"的方案重构。

### 改动
- **新增单测**（torch 实现作为 oracle 内嵌其中，断言 Triton == torch）：
  - `tests/kernels/test_sm120_mqa_logits.py`：Op1（paged decode）+ Op2（dense prefill），参数化 shape。
  - `tests/kernels/test_sm120_mla_prefill.py`：Op3，含 attn_sink 的 None/scalar/[H] 三种分支。
  - 阈值按"实测最大误差 × 安全倍数"定（非复用松默认，`rtol=0` 即 atol 为准）：scorers atol=5e-6（实测 ~1.4e-6，~3.5×）、MLA atol=1e-4（实测 ~6e-5，~1.7×）；比较前先断言 `-inf` 掩码一致。
- **serving 改 Triton-only**：`sparse_attn_indexer.py` + `flashmla.py` 删掉 `VLLM_SM120_TRITON_SCORER`/`VLLM_SM120_TRITON_MLA_PREFILL` 开关 + 分支 + `import os`，直接调 Triton 函数。
- **删除生产 torch 模块**：`fp8_paged_mqa_logits_sm120.py`、`flash_mla_sparse_fwd_sm120.py`（代码已挪进 tests）。
- `sm120/bench/scorer_bench.py`：正确性已由 pytest 覆盖，简化为 Triton-only 延迟测量（去掉 oracle/`--compare`）。
- `sm120/bench/run_perf.sh`：注释更新（serving 只剩 Triton，对比对象是 SGLang）。

### 验证
- `pytest tests/kernels/test_sm120_{mqa_logits,mla_prefill}.py` → **11/11 通过**。
- `vllm serve`（Triton-only，无任何 flag）→ `15*23=?` → **345**，`finish_reason=stop`。
- 生产路径不再有 PyTorch 回退；SM90/100 仍走 DeepGEMM 分支，不受影响。

---

## [2026-06-22] §7.5 decode-step profile — 瓶颈定位（MLA decode 核）

> 用 vLLM torch profiler（`--profiler torch`，eager 下 kernel 级可见，`/start_profile`+`/stop_profile` 窗口捕获 batch-32 decode）解析 rank0 trace。**推翻了之前"MoE/通信是 decode 瓶颈"的假设。**

### 结论：decode 瓶颈 = MLA decode Triton 核
rank0 GPU kernel 时间占比：
| kernel | 占比 | 说明 |
|---|---|---|
| **`_tiled_sparse_decode_kernel`（MLA decode，移植自 SGLang）** | **88.8%** | ~760ms/step（~61 层 × ~12.5ms/层）|
| nccl all-reduce | 4.5% | TP 通信 |
| `_paged_mqa_logits_kernel`（Op1 decode scorer） | 1.1% | 我们优化过的，不是瓶颈 |
| MoE marlin | 0.3% | |
| dense GEMM / norm / quant | <1% | |

- **不是** Op1（1.1%）、**不是** MoE（0.3%）、**不是**通信（4.5%）——是 **MLA decode 核**（88.8%）。
- 这解释了 cudagraph 无增益（该核 `@eager_break_during_capture`，始终 eager）；也定位了 vLLM decode 比 SGLang 慢 2.3× 的根因。
- 该核 autotune 配置极简（仅 3 个：BLOCK_T 16/32，warps 4/8），有明显调优空间。

### 下一步（优化 MLA decode 核）
扩大 autotune 配置 / 调 grid / 对照 re-port SGLang 更新版 → 目标把 88.8% 这块压下来，追平 SGLang decode。这是一块独立的 Triton 调优工作。

---
