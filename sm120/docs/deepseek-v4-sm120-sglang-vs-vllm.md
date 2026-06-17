# 在 sm_120 上跑 DeepSeek-V4-Flash：SGLang 的实现方式与 vLLM 移植方案

> 适用场景：在 **NVIDIA RTX PRO 5000 / 6000（Blackwell，compute capability `12.0` = sm_120）** 上部署 **DeepSeek-V4-Flash** 时，调查 **SGLang 为什么能跑、vLLM 0.23.0 为什么不能**，并给出把同等能力移植进 vLLM 的行动方案。
> 关联文件：`launch.sh`、模型 `config.json`、`./fp8-blockscaled-backends.md`（FP8 线性层 backend 选择，另一堵墙）、SGLang 仓库 `sgl-project/sglang`（main 分支，sm_120 支持在 nightly `lmsysorg/sglang:dev`）。

---

## 0. TL;DR

- **根本差异**：vLLM 写死的那个融合 **DeepGEMM `fp8_einsum` 注意力 o_proj**，在 SGLang 里只是一个**默认关闭**的可选优化（`SGLANG_OPT_FP8_WO_A_GEMM`）。SGLang 在 sm_120 上不打开它，退回**普通 bf16 einsum + 标准 FP8 linear**；再配两个手写的 **sm_120 专用 Triton 核**（MoE + MLA decode）。
- **DeepGEMM 为什么不行**：它依赖 **TMEM / tcgen05**，那是 **SM100 数据中心 Blackwell（B100/B200）** 才有的；桌面/专业卡 **sm_120 没有**。vendored DeepGEMM 只编了 SM90/SM100 内核/PTX。
- **vLLM 的墙不止 o_proj**：稀疏 MLA 后端 (`sparse_mla.py:89`) 和 FlashMLA 内核 (`flashmla.py:69`) 都显式把 sm_120 排除。所以**只补 o_proj 不能让服务跑起来**，只会把崩溃挪到注意力核。
- **现实建议**：要"现在就跑"用 **SGLang nightly**；要在 vLLM 上做，是**周-月级的大型 PR**（≈ 把 SGLang 的 sm_120 内核集搬进 vLLM），建议分层上游化。

---

## 0.5 进度与现状（2026-06-17 更新；**目标已锁定 fp4 checkpoint**）

> **主线 = `/warehouse/DeepSeek-V4-Flash`（fp4 checkpoint）**，对齐 SGLang sm_120 配方。FP8 checkpoint（`...-FP8`）仅作验证 o_proj 用，已弃用。

**确认的事实**（读 fp4 的 `config.json`）：
- fp4 checkpoint 自带 `expert_dtype:"fp4"` + `scale_fmt:"ue8m0"` + `quant_method:"fp8", fmt:"e4m3"`。
- 即：**MoE 专家是 MXFP4（4-bit）**；**注意力/稠密线性层是 FP8 e4m3 + ue8m0 block scale**。（早期文档把 fp4 专家误记成 "FP8 e4m3"，已修正。）

**fp4 上确认的墙序**（每堵都已实测）：

| # | 墙 | 状态 |
|---|---|---|
| ① | **ue8m0 FP8 线性层**（`fused_wqa_wkv` 等）→ CUTLASS `dispatch_scaled_mm` 落空（无 e8m0 kernel） | **当前，Tier 1b** |
| ② | attention o_proj 硬编码 DeepGEMM `fp8_einsum`（`t.dim()==N`） | ✅ **Tier 1 已解决**（`sm120/` 补丁，checkpoint 无关） |
| ③ | MLA indexer 的 DeepGEMM attention 内核（`get_paged_mqa_logits_metadata`，`attention.hpp:219 Unsupported architecture`） | Tier 2（最大） |
| ④ | MXFP4 MoE 运行时（Marlin 能加载，sm_120 上能否跑待验） | Tier 3 |

> 注：fp4 的 MoE 专家走 `Mxfp4MoEMethod` + MARLIN，**加载正常**（日志 `Using 'MARLIN' Mxfp4 MoE backend`），第一堵墙是注意力/稠密线性层，不是 MoE。

---

## 1. 背景：两份 sibling checkpoint

两份 checkpoint，结构不同（详见 `./fp8-blockscaled-backends.md`）：

| Checkpoint | 专家权重 | 注意力/稠密线性层 | vLLM 0.23.0 上的第一堵墙 |
|---|---|---|---|
| `/warehouse/DeepSeek-V4-Flash`（**fp4，当前目标**） | **MXFP4 (FP4)** + ue8m0 scale | FP8 e4m3 + **ue8m0** block scale | 注意力/稠密线性层 ue8m0 → CUTLASS `dispatch_scaled_mm` 落空（墙 ①） |
| `/warehouse/DeepSeek-V4-Flash-FP8`（FP8，已弃用） | FP8 e4m3 + **float32** scale | FP8 e4m3 + float32 scale | 注意力 o_proj 硬编码 DeepGEMM（墙 ②；o_proj 已用此 checkpoint 验证修复） |

> o_proj 修复（Tier 1）是在 FP8 checkpoint 上验证的（因其注意力线性层是 f32-scale，能到达 o_proj），但 o_proj 方案本身 **checkpoint 无关**，fp4 同样适用。fp4 的第一堵墙（ue8m0 线性层，墙 ①）比 o_proj 更靠前。

---

## 2. （历史）o_proj 墙 —— 注意力输出投影硬编码 DeepGEMM【✅ 已解决】

> 本节记录的是 **FP8 checkpoint** 上发现并已修复的 o_proj 墙（墙 ②）。fp4 checkpoint 上它被墙 ①（ue8m0 线性层）挡在前面，但 o_proj 方案 checkpoint 无关，迟早同样需要。修复见 `sm120/`（Tier 1 已完成）。

加 `expert_dtype:fp8`（FP8 checkpoint）后权重能全部加载，崩在预热前向的**注意力输出投影**：

```
nvidia/model.py:851   self.attn(...)
attention.py:364      self._o_proj(o, positions)
nvidia/flashmla.py:43 deep_gemm_fp8_o_proj(...)          ← 两个 CUDA 注意力后端都走它
nvidia/ops/o_proj.py:61  fp8_einsum(...)
utils/deep_gemm.py:303 _fp8_einsum_impl(...)
RuntimeError: Assertion error (deepgemm-src/.../utils/layout.hpp:39): t.dim() == N
```

关键代码事实（vLLM 0.23.0）：

```python
# vllm/models/deepseek_v4/nvidia/flashmla.py:42  （flashinfer_sparse.py:86 同样）
def _o_proj(self, o, positions):
    return deep_gemm_fp8_o_proj(...)   # 无条件，无 else，无架构判断

# vllm/models/deepseek_v4/attention.py:144  （base 类）
def _o_proj(self, o, positions):
    raise NotImplementedError          # 没有非 DeepGEMM 的默认实现
```

`fp8_einsum`（`utils/deep_gemm.py:299`）**绕过** `is_deep_gemm_supported()` 架构门、也绕过 `kernel_config.linear_backend`/`moe_backend`——所以 `--linear-backend marlin`、`--attention-backend ...` 对它都无效。vendored DeepGEMM 只有 SM90/SM100 内核 → sm_120 上 CuTe layout 断言 `t.dim()==N` 崩。

---

## 3. SGLang 在 sm_120 上怎么跑的（三大支柱）

调查来源：`sgl-project/sglang` main 分支（sm_120 支持只在 nightly）。核心策略一句话——**全链路绕开 DeepGEMM 与 ue8m0 硬件路径**。

### 3.1 支柱 ① — 检测 sm_120 并整体关闭 DeepGEMM

```python
# sglang/srt/layers/deep_gemm_wrapper/configurer.py
def _compute_enable_deep_gemm():
    sm = get_device_sm()            # sm_120 → 120
    if (_is_cuda and sm < 90): return False
    if sm == 120: return False      # ← SM120 直接禁用 DeepGEMM（需 TMEM/tcgen05，SM120 没有）
    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()
```

辅助函数（`sglang/srt/utils/common.py`）：`is_sm120_supported`（major==12, CUDA≥12.8）与 `is_sm100_supported`（major==10）**互斥**。
连带后果：**MegaMoE / DeepEP / expert-parallel 全部依赖 DeepGEMM → sm_120 上全关**，只能 TP-only（这也是文档里 RTX PRO 6000 只支持 low-latency/TP-only 的原因）。密集 FP8 线性层的 GEMM 后端由 `fp8_utils._dispatch_auto_backend()` 自动选，sm_120 上落到 **FlashInfer / CUTLASS**（不是 DeepGEMM）。

### 3.2 支柱 ② — MoE 专家：`--moe-runner-backend marlin` 实际跑的是纯 Triton

`marlin` 只是个**路由枚举**，选中 `Mxfp4MarlinMoEMethod`（`layers/quantization/mxfp4_marlin_moe.py`）。其 `process_weights_after_loading`：

```python
if is_sm120_supported():
    # SM120: Skip Marlin repacking (Marlin CUDA kernel produces NaN on SM120)
    layer._dsv4_mxfp4_backend = "sm120_triton"
    return
```

`apply()` 检测到 `"sm120_triton"` 就分发到手写的纯 Triton 核 `layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`（融合 FP4 反量化 + GEMV，float32 累加，针对 sm_120 的 99KB SMEM / 48-warp / 无 TMEM 约束）。

**块 scale 的处理（绕开 e8m0 硬件）**：checkpoint 存的是 e8m0（`float8_e8m0fnu`），加载时一次性转 float32——

```python
layer.w13_weight_scale_inv = Parameter(
    w13_s.view(torch.uint8).view(torch.float8_e8m0fnu).to(torch.float32), ...)
```

运行时 Triton 核里只用 float32（`tl.float32` 累加，bf16 输出），**从不用 e8m0 硬件解释**。这正是 SGLang 能避开 vLLM Variant A 那个 `float8_e8m0fnu` KeyError / CUTLASS dispatch 落空的原因。

### 3.3 支柱 ③ — Attention o_proj 默认就是 bf16（**决定性差异**）

`models/deepseek_v4.py:420-428`，`wo_a` 定义：

```python
_FP8_WO_A_GEMM = envs.SGLANG_OPT_FP8_WO_A_GEMM.get()   # 默认 OFF
self.wo_a = ColumnParallelLinear(
    ...,
    quant_config = quant_config if _FP8_WO_A_GEMM else None,          # 默认 None → bf16
    **({} if _FP8_WO_A_GEMM else {"params_dtype": torch.bfloat16}),  # 默认 bf16
)
```

o_proj 前向（`deepseek_v4.py:996-1019`）：

```python
if _FP8_WO_A_GEMM:                              # 可选优化，默认 OFF；需 SM100/DeepGEMM
    import deep_gemm
    deep_gemm.fp8_einsum("bhr,hdr->bhd", ...)   # ← 融合 DeepGEMM 路径（= vLLM 写死的那个）
else:                                           # 默认（SM120 走这里）
    wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
    o = torch.einsum("tgd,grd->tgr", o, wo_a)   # ← 纯 bf16 einsum，零 DeepGEMM
o, _ = self.wo_b(o.flatten(1))                  # 标准 FP8 RowParallelLinear（GEMM 后端可插拔）
```

`wo_b` 是标准 `RowParallelLinear(quant_config=quant_config)`，FP8 GEMM 后端在 sm_120 自动落到 FlashInfer/CUTLASS。

> **这就是全部秘密**：vLLM 把 SGLang 默认关闭的 `if _FP8_WO_A_GEMM` 分支写死成了唯一路径，且**删掉了 `else`**。

### 3.4 支柱 ④ — MLA decode 用专用 sm_120 Triton 核

`layers/attention/flash_mla_sm120_triton.py`（文件头注明 *"Target: RTX PRO 6000 (SM120, 188 SMs, 99KB SMEM …)"*）：纯 Triton 的 tiled sparse MLA decode 核，手动 FP8→反量化、base-2 online softmax、无 TMEM，CUDA-graph 安全。**不是**数据中心的 FlashMLA C 核（那玩意只 SM90/SM100）。

### 3.5 SGLang vs vLLM 对比

| 环节 | SGLang (sm_120) | vLLM 0.23.0 |
|---|---|---|
| MoE 专家 | Triton `mxfp4_moe_sm120_triton.py`（e8m0→f32） | Marlin 可走（日志见 `Using 'MARLIN' Mxfp4 MoE backend`）——**MoE 不是墙** |
| attention o_proj | **bf16 einsum（默认）** | **硬编码 DeepGEMM `fp8_einsum`，无 else** ← 当前崩溃 |
| MLA decode 核 | Triton `flash_mla_sm120_triton.py` | FlashMLA C 核，门控排除 sm_120 |
| KV cache | bf16 / per-tensor fp8_e4m3 | 强制 `fp8_ds_mla`（ue8m0） |
| DeepGEMM | 全局禁用 | 注意力 + 部分 MoE 路径强依赖 |
| 并行 | TP-only | TP（但注意力栈跑不起来） |

---

## 4. vLLM 0.23.0 在 sm_120 上的墙（完整清单，按 fp4 实测顺序）

| # | 墙 | 位置 | 状态 |
|---|---|---|---|
| 1 | **ue8m0 FP8 线性层** → CUTLASS `dispatch_scaled_mm` 落空（无 e8m0 kernel） | `attention.py:411 fused_wqa_wkv` → `linear.py:582` → `fp8.py:476` → `cutlass.py:324 cutlass_scaled_mm`（影响所有 FP8 block 线性：wqa_wkv/wq_b/wo_b/shared_experts） | **当前，Tier 1b** |
| 2 | o_proj 硬编码 DeepGEMM `fp8_einsum`，无 `else` | `nvidia/flashmla.py:43`→`ops/o_proj.py:61`（FlashMLA+FlashInfer 两后端；base `attention.py:144` 是 `NotImplementedError`） | ✅ **已解决**（Tier 1，`sm120/`） |
| 3 | MLA indexer 的 DeepGEMM attention 内核 `Unsupported architecture` | `v1/attention/backends/mla/indexer.py:613` → `utils/deep_gemm.py:405 get_paged_mqa_logits_metadata`（旁还有 `fp8_fp4_paged_mqa_logits:408`）；均绕过 `is_deep_gemm_supported()` 门 | Tier 2 |
| 4 | 稀疏 MLA 后端能力门 `major in [9,10]` | `sparse_mla.py:89 supports_compute_capability` | sm_120(12) 被排除（墙 3 之后会撞） |
| 5 | FlashMLA sparse 内核门 family 90/100 | `v1/attention/ops/flashmla.py:69 is_flashmla_sparse_supported`（C 核 `_flashmla_C` 只编 SM90/SM100） | 无 sm_120 注意力核（Tier 2） |
| 6 | KV cache 强制 `fp8_ds_mla`（ue8m0） | `attention.py:84` | ue8m0 无 sm_120 核（Tier 2） |
| 7 | 融合 `_C.fused_deepseek_v4_*` 算子 | `attention.py:548/567/581` | sm_120 支持存疑（Tier 2） |
| 8 | MXFP4 MoE 运行时 | `Mxfp4MoEMethod` + MARLIN（加载 OK，运行时待验；SGLang 的 Marlin 在 sm_120 出 NaN，写了 Triton fallback） | Tier 3 |

> ⚠️ 只补 o_proj（墙 2）远不够——fp4 上墙 1 先挡路，之后墙 3~7（MLA 注意力整块）才是最大工程。

---

## 5. vLLM 移植行动方案（分层）

### Tier 1 — o_proj bf16 fallback【✅ 已完成，`sm120/`】

在 FP8 checkpoint 上验证通过（见 `sm120/CHANGES.md`）：给 `nvidia/flashmla.py` + `flashinfer_sparse.py` 的 `_o_proj` 加 `if is_deep_gemm_supported(): <原 DeepGEMM> else: <sm120_o_proj>`，后者在 `ops/o_proj.py` 新增纯 torch 实现（逆向 RoPE + wo_a fp8→bf16 反量化 + grouped einsum + wo_b）。`t.dim()==N` 消失，前向越过 o_proj，撞到墙 ③（MLA）。方案 checkpoint 无关，fp4 同样适用。

### Tier 1b — ue8m0→f32 线性层 scale 转换【进行中，fp4 第一堵墙】

墙 ① 的修法：sm_120 上把 FP8 block 线性层的 scale 从 e8m0 转成 **f32**（`(sf.view(uint8).to(int32)<<23).view(float32)` = `nvidia/model.py:282 _ue8m0_uint8_to_float` 同款），CUTLASS 的 e4m3+f32-block `scaled_mm` 在 Blackwell 上能跑（CUTLASS 只对 **e8m0** 没注册 kernel）。这正是 SGLang 的做法。注入点：`fp8.py Fp8Config.process_weights_after_loading` 的 block_quant 分支末尾，门控 `is_scale_e8m0 and not is_deep_gemm_supported()`——`is_scale_e8m0` 只在 DeepSeek-V4 fp4 为 True，**天然限定在 DeepSeek-V4，不动其他 FP8 模型**。

### Tier 2 — 让 MLA 注意力真能跑（周级，核心工程）

3. **移植/重写 sm_120 MLA decode 核**：参考 SGLang `flash_mla_sm120_triton.py`（纯 Triton，无 TMEM，CUDA-graph 安全）。先 decode-only。墙 ③（`get_paged_mqa_logits_metadata` / `fp8_fp4_paged_mqa_logits`）+ 墙 ⑤（FlashMLA C 核）都在此。
4. **KV cache 去 ue8m0**：sm_120 上把 KV cache 从 `fp8_ds_mla` 退到 bf16 / per-tensor fp8_e4m3（墙 ⑥，改 `attention.py:84` 加 sm_120 分支）。
5. **放开能力门**：`sparse_mla.py:89` → `major in [9,10,12]`；`flashmla.py:69` 加 family 12（墙 ④）。

### Tier 3 — MXFP4 MoE 运行时（周级）

6. fp4 专家当前走 `Mxfp4MoEMethod`+MARLIN（加载 OK）。验证 Marlin 在 sm_120 运行时是否出 NaN；若出，移植 SGLang 的 `mxfp4_moe_sm120_triton.py`（墙 ⑧）。

### 预期管理 & 上游化

- 这是**大型 PR**（≈ 把 SGLang 的 sm_120 内核集搬进 vLLM），生产质量做完是周-月级。建议按 Tier 分多个 PR 上游，每层带 sm_120 CI。
- vLLM 0.23.0 发布于 DeepSeek-V4 sm_120 支持成熟之前，**几乎肯定会自己补 sm_120 路径**——先盯 vLLM release/PR，可能省掉 Tier 2/3 大部分工作。
- **"现在就要跑"**：SGLang nightly（`lmsysorg/sglang:dev`）是唯一现成方案：

  ```bash
  docker run --gpus all --shm-size 32g --ipc=host -p 30000:30000 \
    -v /warehouse:/warehouse \
    lmsysorg/sglang:dev \
    sglang serve --trust-remote-code \
      --model-path /warehouse/DeepSeek-V4-Flash \   # fp4 那份，不是 -FP8
      --tp 8 --moe-runner-backend marlin \
      --mem-fraction-static 0.70 --cuda-graph-max-bs 32 \
      --host 0.0.0.0 --port 30000
  ```

---

## 6. 关键文件:行号速查

### vLLM 0.23.0（`/root/native-vllm/.venv/lib/python3.12/site-packages/vllm/`）
- o_proj 硬编码 DeepGEMM：`models/deepseek_v4/nvidia/flashmla.py:42`、`nvidia/flashinfer_sparse.py:85`
- 融合 o_proj 实现：`models/deepseek_v4/nvidia/ops/o_proj.py:26`（`deep_gemm_fp8_o_proj` → `fp8_einsum:61`）
- recipe 误判 sm_120 为 SM100：`nvidia/ops/o_proj.py:19-23`（`compute_fp8_einsum_recipe`）
- base `_o_proj = NotImplementedError`：`models/deepseek_v4/attention.py:144`
- `wo_a`/`wo_b` 定义：`models/deepseek_v4/attention.py:213`（`quant_config=quant_config`，FP8）/ `:223`
- 稀疏 MLA 能力门：`models/deepseek_v4/sparse_mla.py:89`（`major in [9,10]`）
- FlashMLA sparse 内核门：`v1/attention/ops/flashmla.py:69`（family 90/100）
- KV cache 强制 fp8_ds_mla：`models/deepseek_v4/attention.py:84-87`
- 融合 `_C.fused_deepseek_v4_*`：`models/deepseek_v4/attention.py:548/567/581`
- DeepGEMM 架构门：`platforms/cuda.py`（`support_deep_gemm`，仅 sm_90 + family-100）；`utils/deep_gemm.py:87`
- DeepGEMM kernel 架构：`third_party/deep_gemm/`（仅 SM90/SM100 PTX/tcgen05）

### SGLang（`sgl-project/sglang` main）
- sm_120 检测：`python/sglang/srt/utils/common.py`（`is_sm120_supported`）
- 关闭 DeepGEMM：`python/sglang/srt/layers/deep_gemm_wrapper/configurer.py`（`sm==120 → False`）
- **o_proj bf16 默认路径**：`python/sglang/srt/models/deepseek_v4.py:420-428`（`wo_a` 定义）、`:996-1019`（`else` 分支）、`:138`（`_FP8_WO_A_GEMM`）
- sm_120 MoE Triton 核：`python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- MoE marlin 路由 → sm_120 分支：`python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py`（`_dsv4_mxfp4_backend="sm120_triton"`）
- sm_120 MLA decode Triton 核：`python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- FP8 线性 GEMM 后端自动选择：`python/sglang/srt/layers/quantization/fp8_utils.py`（`_dispatch_auto_backend`）

---

## 7. 探测命令

```bash
# 确认本机 sm_120
uv run python -c "import torch; print(torch.cuda.get_device_capability(0))"   # → (12, 0)

# vLLM DeepGEMM 真实门控（别用裸 import deep_gemm / uv pip show）
uv run python -c "
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.deep_gemm import is_deep_gemm_supported
print('has_deep_gemm', has_deep_gemm())            # True（vendored 可 import）
print('is_deep_gemm_supported', is_deep_gemm_supported())  # False（架构门）
"

# 复现 o_proj 崩溃
bash launch.sh   # 加 expert_dtype:fp8 后，崩在 attention.py:364 _o_proj → deep_gemm layout.hpp:39
```
