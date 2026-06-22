# HANDOFF — sm_120 DeepSeek-V4 enablement → vLLM 源码仓库（editable install）

> 本文件面向**下一阶段**：把 DeepSeek-V4 在 sm_120（RTX PRO 5000/6000 Blackwell）上的 vLLM 支持从「改已安装 wheel」迁移到「在 vLLM 源码仓库里开发、editable install 调试」。`sm120/` 整个目录会被复制到新的 vLLM 源码项目下，作为完整上下文。
> 配套：`README.md`（总览）、`CHANGES.md`（每条改动的逐行记录 + 验证）、`docs/`（调查文档）。

---

## 0. 一句话现状

**已验证可用**：DeepSeek-V4-Flash（**fp4** checkpoint `/warehouse/DeepSeek-V4-Flash`）在 8× RTX PRO 5000（sm_120）上经 vLLM 0.23.0 端到端正确推理——`chat/completions` "用一句话介绍你自己"→连贯中文；"15×23=?"→`345`（算术正确）。当前实现是**对已安装 wheel 的就地补丁**；下一步改成源码树里的正式编辑。

---

## 1. 为什么 sm_120 需要特殊处理（核心约束）

sm_120 是桌面/专业 Blackwell，**没有 TMEM / tcgen05**（那是 SM100 数据中心 B100/B200 才有）。后果——vLLM 0.23.0 里所有依赖这些的内核在 sm_120 上都崩：

| 受影响的路径 | sm_120 上的报错 |
|---|---|
| DeepGEMM（GEMM + attention.hpp 的 einsum/MQA-logits/paged-mqa） | `Unsupported architecture` / `t.dim()==N` |
| FlashMLA C 核（`flash_mla_with_kvcache` decode、`flash_mla_sparse_fwd` prefill） | `only supported on SM90a and SM100f` |
| FlashInfer TRT-LLM FMHA（`TllmGenFmhaRunner`） | `Unsupported architecture`（已实测排除"换 FlashInfer 后端"这条路） |
| CUTLASS `scaled_mm` 对 `float8_e8m0fnu`（ue8m0）block scale | `dispatch_scaled_mm … helper:17`（无 e8m0 kernel） |

所以策略 = **全链路绕开 DeepGEMM / FlashMLA-C / ue8m0**，用 Triton/PyTorch/bf16 替换（对齐 SGLang nightly 的 sm_120 做法）。MoE MXFP4-Marlin 原生支持 sm_120，**无需改**。

---

## 2. 范式切换：wheel 补丁 → 源码编辑（editable install）

| | 旧（当前 sm120/ 做的事） | 新（源码仓库） |
|---|---|---|
| 改在哪 | `.venv/.../vllm/*.py`（已安装 wheel）就地改 | clone vLLM 源码，直接改源 `.py` |
| 生效方式 | 改完重起 vllm；`apply.sh`/`revert.sh` 维护 | `pip install -e .`（editable），改源码即时生效 |
| 复现 | `uv sync` → `apply.sh` | clone → `pip install -e .` → 改源码（改动即本） |
| `apply.sh`/`revert.sh` | 必须 | **不再需要**（保留仅供旧 wheel 环境参考） |

**关键**：改动内容**完全一致**，只是载体从"wheel 补丁"变成"源码编辑"。下面第 3 节给出精确映射。

---

## 3. 改动清单：每条改动 → vLLM 源码路径（迁移用）

> 路径均相对 `vllm/` 包根。`sm120/` 里的 canonical 副本是迁移源。
> - **NEW**：直接把 `sm120/<file>` 复制到源码路径。
> - **EDIT**：若源码也是 0.23.0，可直接 `cp sm120/patched/<f> <src>/<f>`；若是 main（可能已分叉），用 `diff sm120/backups/<f>.orig sm120/patched/<f>` 看改了啥，再手动套用（`CHANGES.md` 有逐条 old→new）。

| # | 源码路径（`vllm/` 下） | 类型 | sm120/ canonical | 作用 |
|---|---|---|---|---|
| 1 | `models/deepseek_v4/nvidia/ops/sm120_o_proj.py` | **NEW** | `sm120_o_proj.py` | o_proj 的 bf16 实现（inv-RoPE + wo_a fp8→bf16 反量化 + grouped einsum + wo_b） |
| 2 | `models/deepseek_v4/nvidia/ops/o_proj.py` | EDIT | `patched/o_proj.py` | +1 行 re-export `sm120_o_proj` |
| 3 | `models/deepseek_v4/nvidia/flashmla.py` | EDIT | `patched/flashmla.py` | +imports；`_o_proj`/`_forward_decode`/`_forward_prefill` 各加 `if is_deep_gemm_supported() else <sm120>` |
| 4 | `models/deepseek_v4/nvidia/flashinfer_sparse.py` | EDIT | `patched/flashinfer_sparse.py` | +imports；`_o_proj` 加 if/else |
| 5 | `models/deepseek_v4/nvidia/flash_mla_sm120_triton.py` | **NEW** | `flash_mla_sm120_triton.py` | MLA **decode** Triton 核（从 SGLang 逐字移植） |
| 6 | `models/deepseek_v4/nvidia/flash_mla_sparse_fwd_sm120.py` | **NEW** | `flash_mla_sparse_fwd_sm120.py` | MLA **prefill**（稠密 bf16 KV 上 gathered sparse attention，PyTorch） |
| 7 | `v1/attention/backends/mla/indexer.py` | EDIT | `patched/indexer.py` | 守卫 `has_deep_gemm()`→`is_deep_gemm_supported()` + import |
| 8 | `model_executor/layers/fp8_paged_mqa_logits_sm120.py` | **NEW** | `fp8_paged_mqa_logits_sm120.py` | 两个 c4 indexer scorer（paged decode + 非 paged prefill），PyTorch 分块 |
| 9 | `model_executor/layers/sparse_attn_indexer.py` | EDIT | `patched/sparse_attn_indexer.py` | +imports；decode/prefill 两处 scorer 调用加 if/else |
| 10 | `model_executor/layers/quantization/fp8.py` | EDIT | `patched/fp8.py` | `process_weights_after_loading` block 分支：ue8m0 scale→f32 |

合计：**4 个新文件 + 6 个编辑**。

---

## 4. 守卫原则（务必保留）

所有 sm_120 分支一律门控 `is_deep_gemm_supported()`（`vllm/utils/deep_gemm.py`）：
- sm_120 → `False`（`support_deep_gemm()` 只认 sm_90 + family-100）→ 走新路径。
- SM90/SM100 → `True` → 走原 DeepGEMM/C 核路径，**行为完全不变**。

迁移到源码后保持同样门控，便于上游化（sm_120 作为 SM90/100 之外的受支持分支）。

---

## 5. 在源码树里重建（editable install）步骤

```bash
# 1) 拿到 vLLM 源码（0.23.0 tag 最稳；main 需按第 3 节重新对齐）
git clone https://github.com/vllm-project/vllm.git && cd vllm && git checkout v0.23.0-release   # 或 main

# 2) editable 安装（按仓库 README；通常用 Python 3.12 + 指定 CUDA）
pip install -e .          # 或 uv pip install -e .；首次会编译 _C 扩展，耗时

# 3) 放入 4 个新文件（从复制过来的 sm120/ 取）
cp <sm120/>/sm120_o_proj.py             vllm/models/deepseek_v4/nvidia/ops/
cp <sm120/>/flash_mla_sm120_triton.py   vllm/models/deepseek_v4/nvidia/
cp <sm120/>/flash_mla_sparse_fwd_sm120.py vllm/models/deepseek_v4/nvidia/
cp <sm120/>/fp8_paged_mqa_logits_sm120.py vllm/model_executor/layers/

# 4) 套用 6 处编辑：0.23.0 可直接覆盖；main 用 diff 手动套（见第 3 节）
cp <sm120/>/patched/{o_proj,flashmla,flashinfer_sparse,fp8,indexer,sparse_attn_indexer}.py <对应源码路径>
#   注意：patched/*.py 是 0.23.0 基线上改的；若源码已变，先 diff sm120/backups/<f>.orig sm120/patched/<f> 看改动再手动套

# 5) 验证 + 起服务
python -c "from vllm.utils.deep_gemm import is_deep_gemm_supported as f; print('sm120 gate:', f())"  # 期望 False
vllm serve /warehouse/DeepSeek-V4-Flash --tp 8 --kv-cache-dtype fp8 --gpu-memory-utilization 0.9
# 起来后测：
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash","messages":[{"role":"user","content":"15*23=?"}],"max_tokens":16}'
```

---

## 6. editable 模式下的调试要点

- 改源码 `.py` → 重起 `vllm serve` 即生效（editable 不需要重装）。
- 想快速迭代可加 `--enforce-eager`（跳过 CUDA graph 捕获，省 ~12min 启动；代价是慢）。
- 几个 sm_120 算子跑在 `breakable_cudagraph` 的 **eager 段**（`_o_proj`、indexer scorer、MLA prefill/decode）——**可以直接 print / pdb**，不受 graph 捕获限制。
- Triton 核（MLA decode、attention 里 vLLM 自带的）首次 JIT 慢，缓存在 `~/.triton`；清缓存可强制重编。
- `is_deep_gemm_supported()` 是 `@functools.cache`，改不了架构判断；要临时强制走某分支，monkeypatch 它或在分支里加临时 print。

---

## 7. 待办（按优先级）

1. ✅ **吞吐优化（已完成 2026-06-18）**：c4 indexer scorer（2 个）+ MLA prefill 已全部 Triton 化（Op1 24.6×、Op2 26.2×、Op3 5.63×），均门控 `is_deep_gemm_supported()` 的 else + PyTorch 回退，长上下文压测通过。**实测：cudagraph 对 sm120 路径无增益**——三核 + MLA 经 `@eager_break_during_capture` 始终 eager，且 MoE/通信也大概率未进 graph，故 cg≈eager（decode 36.6 tok/s、prefill 1669 tok/s）。详见 `CHANGES.md`。
2. ✅ **正确性（部分完成 2026-06-18）**：GSM8K Pass@1 **0.965**（200 题，0 invalid）。剩余：对照参考（SGLang nightly / fp8 baseline）做系统长上下文 + eval。
3. **MoE 运行时**：Marlin MXFP4 跑通无 NaN，但吞吐/长上下文未压测。
4. **prefill 大上下文内存**：MLA prefill 已改 Triton（流式，不再整体 materialize `[Tq,topk,D]`），但更大 topk / 更长序列仍未压测。
5. ⏳ **decode-step profile（待办，暂缓）**：profile 一个 decode step，钉死 ~864ms ITL 的去向（MoE 专家计算 / TP all-reduce / eager-break attention / dense GEMM 各占多少）→ 定位 sm120 真正吞吐瓶颈、判断 MoE/attention 能否做成 graph-safe 以再提速。**由用户标记暂缓，后续处理。**
6. ✅ **清理 torch 服务分支（已完成 2026-06-22）**：PyTorch 三核实现负载下不可用（EngineDeadError），已从生产代码移除——serving 改 Triton-only（删 `VLLM_SM120_TRITON_*` 开关/分支、删两个生产 torch 模块）；torch 实现作为**正确性 oracle 移入 tests**（`tests/kernels/test_sm120_{mqa_logits,mla_prefill}.py`，11/11 通过）。详见 CHANGES.md。

> **范围说明（2026-06-18）**：本路线专注 **v0.23.0 基线**上的开发与优化；"上游对齐 / 上游化到 main" 已明确**移出范围**（不再追求上游化为门控分支）。

---

## 8. 关键技术上下文（避免重新推导）

- **四堵墙 + 解法**：① FP8 线性层 ue8m0→CUTLASS 落空（解：fp8.py ue8m0→f32）；② o_proj 硬编码 DeepGEMM（解：bf16 fallback）；③ MLA decode+prefill + 2× indexer（解：Triton decode + PyTorch prefill/scorer + indexer guard flip）；④ MoE MXFP4-Marlin（无需改）。详见 `docs/deepseek-v4-sm120-sglang-vs-vllm.md` §4。
- **KV 布局兼容**：vLLM 的 `fp8_ds_mla` 页内布局 = SGLang DSv4 布局（每 token 576B 数据 = 448B fp8 nope ‖ 128B bf16 rope，外加 8B/token ue8m0 scale）。所以 SGLang 的 MLA decode Triton 核**逐字可移植**（`flash_mla_sm120_triton.py`）。
- **indexer cache 布局差异（坑）**：vLLM indexer cache 是**逐 token** `[num_blocks, block_size, 1, D+4]`（128B 值 + 4B scale 交错），而 SGLang torch scorer 假设**块级分割**（所有值在前、所有 scale 在后）。移植 scorer 时按 token 切分，不能照搬。
- **内存坑**：indexer scorer 一开始把 `[Bn, max_pages, block_size, D]` 整个 materialize → 64GiB OOM（max_model_len=1M）。改成分块流式 gather。
- **前向顺序**：decode = ① c4 indexer 打分 → ② MLA decode；prefill = ① c4 prefill indexer → ② MLA prefill。indexer 先跑。
- **attn_sink 语义**：作为"汇聚 token"加进 softmax 分母（不贡献分子），即 `denom = Σexp(score) + exp(sink)`，输出按比例缩放。
- **SGLang 参考位置**（`sgl-project/sglang` main）：o_proj bf16（`models/deepseek_v4.py:996-1019`）、MLA decode Triton（`layers/attention/flash_mla_sm120_triton.py`）、indexer torch scorer（`layers/attention/dsv4/indexer.py` 的 `fp8_paged_mqa_logits_torch_sm120`）、sm_120 检测（`utils/common.py:is_sm120_supported`）、全局关 DeepGEMM（`layers/deep_gemm_wrapper/configurer.py`）。

---

## 9. sm120/ 文件清单（复制到新项目后各是什么）

| 文件/目录 | 作用 | 源码树模式下还需？ |
|---|---|---|
| `HANDOFF.md` | 本文件 | 参考 |
| `README.md` | 总览 | 参考 |
| `CHANGES.md` | 逐条改动记录 + 验证 | ⭐ 迁移时的逐条依据 |
| `docs/` | 调查文档（SGLang vs vLLM、FP8 backend） | 参考 |
| `launch.sh` | 起服务命令 | 可用（绝对 checkpoint 路径，任意 cwd 可跑） |
| `sm120_o_proj.py` / `flash_mla_sm120_triton.py` / `flash_mla_sparse_fwd_sm120.py` / `fp8_paged_mqa_logits_sm120.py` | 4 个新模块 canonical 源 | ⭐ 复制进源码树 |
| `patched/*.py`（6 个） | 已打补丁版（0.23.0 基线） | ⭐ diff/覆盖进源码树 |
| `backups/*.orig`（6 个） | 0.23.0 原始版 | diff 基线 |
| `apply.sh` / `revert.sh` | wheel 模式维护脚本 | 源码模式下**不需要**，保留备查 |

---

## 10. 环境与依赖（必须带到新环境）

- **硬件**：8× NVIDIA RTX PRO 5000 72GB，`compute_capability=(12,0)`=sm_120。
- **软件（验证过的版本）**：vLLM 0.23.0、torch 2.11.0+cu130、CUDA 13.0、Python 3.12、FlashInfer 0.6.12、triton 3.6、tilelang 0.1.9、transformers 5.12.1。
  - 注意：vLLM 默认 `support_deep_gemm()` 只认 sm_90 + family-100，**不覆盖 sm_120**——这正是本套补丁存在的原因。
- **模型**：`/warehouse/DeepSeek-V4-Flash`（fp4；`config.json`: `expert_dtype:"fp4"`, `scale_fmt:"ue8m0"`, `quant_method:"fp8"`, `fmt:"e4m3"`；`index_head_dim=128`, `index_n_heads=64`, `index_topk=512`, `head_dim=512`, `o_lora_rank=1024`）。**不需要任何 config override**。
- **另一个 checkpoint** `/warehouse/DeepSeek-V4-Flash-FP8`（FP8 专家，缺 `expert_dtype`）仅用于早期验证 o_proj，已弃用，主线用 fp4。
- 起服务参数：`--tensor-parallel-size 8 --kv-cache-dtype fp8 --gpu-memory-utilization 0.9`（详见 `launch.sh`）。

---

## 11. 给接手者的一句话

整套 sm_120 支持是**4 个新文件 + 6 处门控编辑**，核心是把 DeepGEMM/FlashMLA-C/ue8m0 三条 sm_120 走不通的路换成 Triton/PyTorch/bf16（对齐 SGLang）。功能已验证正确；剩余是**性能优化 + 上游对齐**。迁移到源码树后，按第 3 节把改动落成正式编辑，`is_deep_gemm_supported()` 守卫保持不变即可。
