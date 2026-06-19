# sm120 DeepSeek-V4 使能:架构走读与上下文

> 本文沉淀 sm120 分支(`v0.23.0-sm120`,自 `21a7715f` 起)的**架构走读 + 上下文**,补齐现有文档没有或不系统的部分:**上游重构背景、完整调用栈、术语表、SGLang 出处档位**。
> 与现有文档分工:
> - [`HANDOFF.md`](../HANDOFF.md) —— 迁移到源码树 / 范围 / 待办(操作面)。
> - [`CHANGES.md`](../CHANGES.md) —— 每条改动的逐行记录 + 验证(实现面)。
> - [`deepseek-v4-sm120-sglang-vs-vllm.md`](deepseek-v4-sm120-sglang-vs-vllm.md) —— "四堵墙"调查(SGLang vs vLLM)。
> - 本文 —— **为什么这么改(上游 RFC)+ 数据怎么流(调用栈)+ 名词指什么(术语表)**。

---

## 0. TL;DR

- sm_120(RTX PRO 5000/6000,Blackwell,CC 12.0)没有 TMEM/tcgen05,凡硬依赖 DeepGEMM / FlashMLA-C / ue8m0 的路径都崩。
- 本分支把它们换成 **Triton / PyTorch / bf16**(对齐 SGLang nightly)。改动 = **7 个新文件 + 6 处门控编辑**,全部门控 `is_deep_gemm_supported()`(sm120→False),**SM90/SM100 行为不变**。
- 触碰面集中在 **attention + 1 处共享 FP8 修复**(`fp8.py` ue8m0→f32);**MoE(MXFP4-Marlin)一行没改,原生跑通**。
- 端到端正确(GSM8K Pass@1 **0.965**);吞吐上 3 个热点算子(Op1/Op2/Op3)已 Triton 化(24.6× / 26.2× / 5.63×)。**遗留**:对照参考实现的系统回归、MoE/长上下文压测。

---

## 1. 上游背景:DeepSeek-V4 为什么在 `vllm/models/` 下

vLLM 传统上组网代码都在 `vllm/model_executor/models/*.py`(扁平)。**DeepSeek-V4 是个例外**,住在 `vllm/models/deepseek_v4/`。这不是随手放的,而是一次**正式重构**的先导。

### 1.1 总纲:RFC #42770「Changes in vLLM Model Development」

- 出处:<https://github.com/vllm-project/vllm/issues/42770>(WoosukKwon,2026-05-15,pinned)。
- 迁移 PR:`#43004 [Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N]`(分支 `woosuk/dsv4-iso`,iso=hardware isolation),及其 `[2/N]#43039` / `[3/N]#43073` / `[4/N]#43077`。`vllm/model_executor/models/registry.py` 为此加了 `_resolve_module_name`,允许 registry 条目用全限定 `vllm.` 路径(目前仅 V4 的 2 个条目用到)。

### 1.2 RFC 的两条动机 + 三大支柱

**动机**:① 过去太依赖抽象/编译 pass 而非直接改模型代码做融合,代码又晦涩又难优化;② "一份模型定义跑所有硬件"不现实,不同硬件偏好不同融合;③ AI coding agent 让手写融合/算子变容易,且在**原始模型代码**上工作最好。

**三大支柱**:
1. **去掉 full-graph `torch.compile`**(保局部融合,弃整图编译)→ 把"big fused ops"直接写在模型代码里;弃用 CustomOp / vLLM IR 的进一步发展;用 piecewise(breakable)CUDA graph 替代编译器版(见 `#42304`)。
2. **硬件分离 / 模型隔离** ← V4 目录结构的来源。按**厂商**拆分模型代码(`models/<arch>/{nvidia,amd,xpu}/model.py + kernels/ + tests/`),各自独立演进,`__init__.py` 按 `current_platform` 分派。DeepSeek-V4 是这套布局的首个住户,目前唯一。
3. **清晰的模型接口**:vLLM 自有 `ModelConfig`(HF config ∪ model 协议)与 `Model` ABC(模型自管状态、`prepare_inputs`/`forward`)。

### 1.3 对 sm120 工作的意义

本分支的三 op 手写 Triton + `is_deep_gemm_supported()` 运行期分派、sm120 专属 kernel 放进 `vllm/models/deepseek_v4/nvidia/`——**正是支柱 ①(手写融合 + 分派)和 ②(硬件隔离)的实践**。sm_120(无 TMEM/tcgen05,DeepGEMM/FlashMLA-C 走不通)是"不同硬件要不同融合"的活样本。

> 相关子 RFC:[`#43224 Porting compiler fusions to manual fusion`](https://github.com/vllm-project/vllm/issues/43224);piecewise CUDA graph:[`#42304`](https://github.com/vllm-project/vllm/pull/42304)。

---

## 2. 本分支工作脉络(自 `21a7715f`)

7 个提交,主线 = **地基 → 三个热点算子 Triton 化 → 收尾验证**:

| 提交 | 内容 |
|---|---|
| `2a7ff72af` | **地基**:wheel 补丁迁入源码树,4 新模块 + 6 门控编辑,端到端跑通 |
| `d40187b32` | **Op1** c4 分页 **decode** scorer → Triton(~24×) |
| `b9beceb45` | **Op2** c4 密集 **prefill** scorer → Triton(~26×) |
| `edbae01dc` | **Op3** MLA **prefill** gathered-sparse attn → Triton(~5.6×) |
| `9d4f065f4` | §7.1 收尾:autotune 评估(弃用)+ 长上下文压测(通过) |
| `76116f5cc` | Docs:§7.1 标完成;上游化移出 roadmap |
| `169c56406` | GSM8K 正确性(0.965)+ serving 吞吐 harness(Triton vs torch) |

### 2.1 范围界定:动了什么、没动什么

- **动了**:attention 侧(o_proj / MLA decode / MLA prefill / 2× indexer scorer + 它们的 Triton 版) + **1 处共享 FP8 修复**(`fp8.py` ue8m0→f32)。详见 §4。
- **没动**:`MoE` / `fused_moe` / `marlin` / `mxfp4` 一行没改。MXFP4-Marlin 原生支持 sm120,跑通无 NaN。
- **完备性分档**([`HANDOFF.md`](../HANDOFF.md) §7):
  - **功能(能跑 + 正确)**:✅ 端到端 serve 正常、GSM8K 0.965、无 NaN。
  - **吞吐/长上下文压测**:❌ 部分未做 —— MoE 吞吐/长上下文(§7.3)、对照 SGLang/fp8 baseline 的系统回归(§7.2)、prefill 更大 topk/更长序列内存(§7.4)仍在待办。

---

## 3. 术语表:c4 / c4a / c128a / swa / compress_ratio

命名核心 = `compress_ratio`(`config.compress_ratios[layer_id]`,**每层一个值**,取 1/4/128),即该层"旧 KV"的压缩比。每层 = 一个 SWA 滑窗 + 一个压缩池。

| 缩写 | = | 含义(代码依据) |
|---|---|---|
| **C / c** | **C**ompression ratio | KV 压缩比 |
| **C4A** | compress_ratio == 4 | 每 4 个 KV token 压成 1。**唯一带 indexer 的层**:`attention.py:245` "Only C4A uses sparse attention and hence has indexer" |
| **C128A** | compress_ratio == 128 | 每 128 个 KV token 压成 1(粗粒度)。**不跑 indexer**,topk 在 metadata 构建时预算:`sparse_mla.py:124` |
| **SWA** | **S**liding **W**indow **A**ttention | compress_ratio ≤ 1:最近滑窗,全分辨率,无 compressor、无 indexer:`attention.py:499` |
| **c4 indexer / c4 scorer** | C4A 层的稀疏选择器 | 在 C4A 压缩池上打分 + 选 top-K 的核 = Op1(decode)/Op2(prefill) |
| **Lightning Indexer** | 整套稀疏选择机制名 | DeepSeek V3.2/V4 的稀疏注意力选择器(`attention.py:688` logger) |
| **compressor** | `DeepseekCompressor` | 把原始 KV 压成压缩 KV 的模块(RMSNorm + rope + N→1 合并) |

> ⚠️ **"A" 的展开**:代码注释里**没有显式说明**(全树无 `C4A = ...`),只是一致用作层类型后缀(C4A / C128A)。从语境推断 = "该压缩比下的注意力层变体"——这是推断,非代码明写。

**分层稀疏 KV 架构**(每层):
```
单层 KV = { SWA 窗口(最近 N token,全分辨率) }
        + { 压缩池(更早 token,按本层 compress_ratio 压缩) }
            ├─ C4A(ratio 4)  → 压缩池细 → indexer(Op1/Op2)运行时打分选 topk
            ├─ C128A(ratio 128)→ 压缩池粗 → topk 预算,不跑 indexer
            └─ SWA-only(≤1) → 无压缩池
然后 MLA 注意力在 [SWA 全分辨率 + topk 选中的压缩块] 上做 gathered sparse attention
       └─ = Op3(prefill) / flash_mla_sm120_triton(decode)算的东西
```

数据流:**compressor 造压缩 KV → indexer(仅 C4A)打分选 topk(Op1/Op2)→ MLA 注意力(Op3 / decode 核)在 SWA + 选中块上算注意力**。

---

## 4. 完整调用栈(sm120 全部触点)

最上层止于 [`vllm/models/deepseek_v4/nvidia/model.py`](../../../vllm/models/deepseek_v4/nvidia/model.py)。`★` = sm120 专属分支,`✓` = 未改但原生可用。

### 阶段 0:权重加载(一次性,喂给前向的 GEMM)
```
Fp8LinearMethod.process_weights_after_loading          quantization/fp8.py (~L395, block_quant 分支)
└─ if is_scale_e8m0 and not is_deep_gemm_supported():  ★ EDIT#12 fp8.py
   └─ weight_scale_inv: float8_e8m0fnu ─bit-trick→ float32
      (view uint8 → int32 <<23 → view f32; format_ue8m0=False)
      → 所有 FP8 线性层落地为 e4m3+f32,CUTLASS scaled_mm 可分派
```
让**所有 FP8 线性层**的 `scaled_mm` 在 sm120 可用(attention 的 `fused_wqa_wkv`/`wq_b`/`wo_a`/`wo_b` 在前向都调它)。**MoE 是 fp4/Marlin,不走此路**。只影响 DeepSeek-V4 fp4 on sm120。

### 阶段 1:每步 attention metadata 构建(前向之前)
```
DeepseekV32IndexerMetadataBuilder (build)
└─ if current_platform.is_cuda() and is_deep_gemm_supported():  ★ EDIT#10 indexer.py:610
     (原 has_deep_gemm() —— sm120 上 DeepGEMM"装了但不支持",必须翻成 is_deep_gemm_supported())
     sm120 → False → 跳过 get_paged_mqa_logits_metadata()
```

### 阶段 2:每层前向(主体)
```
DeepseekV4ForCausalLM.forward                          model.py:1305
└─ DeepseekV4Model.forward                             model.py:996
   └─ DeepseekV4DecoderLayer.forward                   model.py:803
      ├─ input_layernorm                               ✓ native
      ├─ self.attn = DeepseekV4FlashMLAAttention       model.py:744 (_select_dsv4_attn_cls 默认选它;:716)
      │  └─ DeepseekV4Attention.forward                attention.py:318
      │     ├─ attn_gemm_parallel_execute(hidden)      attention.py:336
      │     │   └─ fused_wqa_wkv/wq_b/wo_a/wo_b (FP8 线性层) → scaled_mm   ★(受益于 fp8.py)
      │     ├─ fused_q_kv_rmsnorm                      ✓ native
      │     ├─ attention_impl(...) (@eager_break)      attention.py:351 / def :427
      │     │   ├─ [仅 C4A 层] indexer(...) aux stream  attention.py:464
      │     │   │  └─ DeepseekV4Indexer.forward        attention.py:766
      │     │   │     ├─ wq_b + fused_indexer_q_rope_quant (并行)
      │     │   │     ├─ compressor(...) (并行)
      │     │   │     └─ indexer_op(...) = SparseAttnIndexer   attention.py:800 / NEW :746
      │     │   │        └─ torch.ops.vllm.sparse_attn_indexer  sparse_attn_indexer.py:521
      │     │   │           └─ sparse_attn_indexer() (@eager_break)  :99
      │     │   │              ├─ has_prefill → scorer   :260  ★ EDIT#11 + Op2
      │     │   │              │   fp8_mqa_logits_sm120_triton | oracle fp8_mqa_logits_sm120
      │     │   │              ├─ has_decode  → scorer   :365  ★ EDIT#11 + Op1
      │     │   │              │   fp8_paged_mqa_logits_sm120_triton | oracle fp8_paged_mqa_logits_sm120
      │     │   │              └─ top_k_per_row_* / persistent_topk → topk_indices_buffer
      │     │   └─ forward_mqa(q,kv,positions,out)      flashmla.py:99
      │     │      ├─ prefill tokens → _forward_prefill  flashmla.py:157 / def :283  ★ EDIT#8
      │     │      │   └─ flash_mla_sparse_fwd_sm120_triton :406 (Op3) | oracle flash_mla_sparse_fwd_sm120
      │     │      └─ decode tokens  → _forward_decode   flashmla.py:167 / def :176  ★ EDIT#8
      │     │          └─ flash_mla_sparse_decode_triton :269 (NEW,地基即 Triton)
      │     │             (else: flash_mla_with_kvcache,SM90/100)
      │     └─ _o_proj(o, positions)                    attention.py:364
      │        └─ sm120_o_proj(...)                     ★ NEW#1 + EDIT#2/flashinfer  flashmla.py:78
      │           (else: deep_gemm_fp8_o_proj,SM90/100)
      ├─ post_attention_layernorm                      ✓ native
      └─ mlp                                           ✓ 整块未碰
          ├─ [稠密层] DeepseekV4MLP                      ✓ native
          └─ [MoE 层] DeepseekV4MoE → FusedMoE
             └─ experts: MXFP4-Marlin GEMM              ✓ 原生支持 sm120,跑通无 NaN
```

### 全部 12 个触点 → 触发位置 + 门控

| # | 文件 | 类型 | 触发处 | sm120 门控 |
|---|---|---|---|---|
| 1 | `nvidia/ops/sm120_o_proj.py` | NEW | 阶段2 `_o_proj` | `not is_deep_gemm_supported()` |
| 2 | `nvidia/ops/o_proj.py` | EDIT | re-export(#1 用) | — |
| 3 | `nvidia/flash_mla_sm120_triton.py` | NEW | 阶段2 `_forward_decode` | `not is_deep_gemm_supported()` |
| 4 | `nvidia/flash_mla_sparse_fwd_sm120.py` | NEW | 阶段2 `_forward_prefill`(Op3 oracle) | `not _USE_TRITON_MLA_PREFILL` |
| 5 | `layers/fp8_paged_mqa_logits_sm120.py` | NEW | 阶段2 indexer(Op1/Op2 oracle) | `not _USE_TRITON_SCORER` |
| 6 | `layers/fp8_mqa_logits_triton.py` | NEW | 阶段2 indexer(Op1/Op2 Triton) | `_USE_TRITON_SCORER` |
| 7 | `nvidia/flash_mla_sparse_prefill_triton.py` | NEW | 阶段2 `_forward_prefill`(Op3 Triton) | `_USE_TRITON_MLA_PREFILL` |
| 8 | `nvidia/flashmla.py` | EDIT | 阶段2(`_o_proj`/decode/prefill) | `not is_deep_gemm_supported()` |
| 9 | `nvidia/flashinfer_sparse.py` | EDIT | 阶段2 `_o_proj`(FlashInfer 后端) | `not is_deep_gemm_supported()` |
| 10 | `v1/attention/backends/mla/indexer.py` | EDIT | 阶段1 metadata | `is_deep_gemm_supported()`(原 `has_deep_gemm()`) |
| 11 | `layers/sparse_attn_indexer.py` | EDIT | 阶段2 indexer 两处 scorer | `not is_deep_gemm_supported()` |
| 12 | `layers/quantization/fp8.py` | EDIT | 阶段0 权重加载 | `is_scale_e8m0 and not is_deep_gemm_supported()` |

> env 开关:`VLLM_SM120_TRITON_SCORER`(默认 1,Op1+Op2)、`VLLM_SM120_TRITON_MLA_PREFILL`(默认 1,Op3)。硬件门控统一 `is_deep_gemm_supported()`(sm120→False)。

---

## 5. 三个热点算子(Op1/Op2/Op3)

| Op | 角色 | 入口函数 | 文件 | 触发条件 | 加速比(vs PyTorch oracle) |
|---|---|---|---|---|---|
| **Op1** | c4 **分页 decode** scorer | `fp8_paged_mqa_logits_sm120_triton` | `model_executor/layers/fp8_mqa_logits_triton.py` | C4A 层 + decode + `not is_deep_gemm_supported()` | 1M/B=8: 28.4ms→1.15ms(**24.6×**) |
| **Op2** | c4 **密集 prefill** scorer | `fp8_mqa_logits_sm120_triton` | 同上 | C4A 层 + prefill + `not is_deep_gemm_supported()` | M=4096,N=32768: 230.8ms→8.8ms(**26.2×**) |
| **Op3** | MLA **prefill** gathered-sparse attn | `flash_mla_sparse_fwd_sm120_triton` | `models/deepseek_v4/nvidia/flash_mla_sparse_prefill_triton.py` | prefill + `not is_deep_gemm_supported()` | Tq=2048: 53.5ms→9.5ms(**5.63×**) |

**数据依赖**:`indexer(Op1/Op2) → topk_indices_buffer → MLA prefill(Op3)`,三者串在同一个 `attention_impl` 调用里(indexer 在 aux stream,`forward_mqa` 在默认 stream 消费)。**只有 C4A 层有 indexer**,所以 Op1/Op2 只在 C4A 层触发。

---

## 6. `flash_mla_sm120_triton.py` 详解(sm120 的 MLA decode 核)

**作用**:sm120 上 MLA 的 decode 注意力核本身。SM90/100 同位置是 DeepGEMM `flash_mla_with_kvcache`;sm120 无可用 MLA decode 核(DeepGEMM FlashMLA、FlashInfer TRT-LLM FMHA 都 `Unsupported architecture`),所以这是 CHANGES.md §90 "墙 ③" 的关键一块,也是让 decode 能跑的最大单块。

**调用点**:`flashmla.py:_forward_decode:269`,消费 Op1 选出的 topk。

**内部结构**(`flash_mla_sm120_triton.py`):
- `_tiled_sparse_decode_kernel`(`:51`,JIT,grid `(num_decode_tokens, padded_heads)`):tile 化(`BLOCK_T` 个 KV 一次 gather),同一 paged buffer 的三种 typed view 做核内 FP8 反量化 —— fp8 view(448 维 nope 值)、uint8 view(ue8m0 scale 字节 → `exp2(scale-127)`)、bf16 view(64 维 rope 值);`kv_nope = fp8 * exp2(scale-127)`,QK = Σ(q·k_nope)+Σ(q·k_rope),**online softmax** 累加 V;输出 `[448 nope ‖ 64 rope]` bf16 + LSE。保留 `@triton.autotune`(`:42`)。
- `_run_triton_sparse_decode`(`:179`):三种 view 装配 + 启动。
- `flash_mla_sparse_decode_triton`(`:276`,公开入口):**核跑两遍** —— 一遍 SWA 本地窗(`k_cache`),一遍 C4A 压缩稀疏块(`extra_k_cache`,topk 来自 indexer);`_merge_partial_attn`(`:244`)按 LSE 合并两段 partial attention;`_apply_attn_sink`(`:261`)把 sink 折进分母。

> 与 prefill 的分工:decode 用本核(每 token 独立 query + gathered 稀疏 KV);prefill 用 Op3(`flash_mla_sparse_fwd_sm120_triton`)。

---

## 7. SGLang 出处档位(逐字/改编/镜像/自研)

| 模块 | 出处 | 档位 | 证据 |
|---|---|---|---|
| `flash_mla_sm120_triton.py`(MLA decode) | SGLang `.../attention/flash_mla_sm120_triton.py` | **🟥 逐字移植** | 头注释 `:2` "Ported verbatim from SGLang";CHANGES.md §97 "逐字移植(Apache-2.0)" |
| `fp8_paged_mqa_logits_sm120.py`(c4 scorer oracle) | SGLang `fp8_paged_mqa_logits_torch_sm120` | **🟧 改编** | 头注释 `:2` "Adapted from SGLang";`:5-8` 差异(逐 token vs 块级切分);CHANGES.md §129 |
| `sm120_o_proj.py`(o_proj bf16) | SGLang 默认 o_proj 算法 | **🟨 镜像算法,自实现** | `:6` "Mirrors SGLang's default o_proj" |
| `flash_mla_sparse_fwd_sm120.py`(MLA prefill oracle) | 无 | **🟩 原创** | 头注释无 SGLang 字样;CHANGES.md §118 未标来源 |
| `fp8_mqa_logits_triton.py`(Op1/Op2 Triton scorer) | 无 | **🟩 原创** | 提交 d40187b32/b9beceb45 描述设计;无来源标注 |
| `flash_mla_sparse_prefill_triton.py`(Op3 Triton prefill) | 无 | **🟩 原创** | 提交 edbae01dc 含"3 个 dead-end"设计记录;无来源标注 |

- **MLA decode 能逐字移植**:因 DSv4 页内布局(每 token 576B = 448B fp8 nope ‖ 128B bf16 rope,+8B/token ue8m0 scale)== vLLM `fp8_ds_mla` == SGLang 布局(CHANGES.md §128)。
- **c4 scorer 不能照搬**:vLLM indexer cache 是**逐 token** `[num_blocks,block_size,1,D+4]`(128B 值 + 4B scale 交错),SGLang 假设**块级分割**(所有值在前、所有 scale 在后);算法借自 SGLang,gather/反量化按 vLLM 布局重写(CHANGES.md §129)。
- **策略层对齐**:sm120 检测 + 全局关 DeepGEMM 的思路参照 SGLang(`utils/common.py:is_sm120_supported`、`deep_gemm_wrapper/configurer.py`),但 vLLM 用自己的 `is_deep_gemm_supported()` 门控,未抄检测/配置代码。

> ⚠️ **上游化 attribution**:严格做 license 标注的是 `flash_mla_sm120_triton.py`(逐字移植,Apache-2.0,CHANGES.md 已记 SGLang 路径);`fp8_paged_mqa_logits_sm120.py` 建议也标 adapted-from。其余自研模块无此要求。注意 [`HANDOFF.md`](../HANDOFF.md) §7 已注明本路线**不追求上游化到 main**。

---

## 8. 范围与遗留

| 部分 | 功能(能跑+正确) | 吞吐/长上下文压测 |
|---|---|---|
| Attention(补的算子) | ✅ 端到端正确 | ✅ Op1/2/3 已 Triton 化(§7.1 完成) |
| **MoE MXFP4-Marlin** | ✅ 跑通无 NaN、端到端正确 | ❌ 吞吐/长上下文未压测(§7.3) |
| 正确性回归(全局) | ⚠️ smoke-test + GSM8K **0.965** | ❌ 对照 SGLang/fp8 baseline 系统回归未做(§7.2) |
| prefill 大上下文内存 | ✅ 流式重写,12k 召回通过 | ❌ 更大 topk/更长序列未压测(§7.4) |

**结论**:sm120 上 DSv4 "能推理且结果对"这条线闭环;"跑得多快、多大上下文不崩、和参考实现逐 token 对齐"这条线,MoE 和全局回归仍压在待办。

---

## 9. 关键参考

- **上游 RFC**:[#42770 Changes in vLLM Model Development](https://github.com/vllm-project/vllm/issues/42770)、[#43004 V4 迁移 1/N](https://github.com/vllm-project/vllm/pull/43004)、[#43224 手写融合](https://github.com/vllm-project/vllm/issues/43224)、[#42304 breakable CUDA graph](https://github.com/vllm-project/vllm/pull/42304)。
- **本仓**:[`HANDOFF.md`](../HANDOFF.md)(迁移/范围/待办)、[`CHANGES.md`](../CHANGES.md)(逐条改动 + 验证)、[`deepseek-v4-sm120-sglang-vs-vllm.md`](deepseek-v4-sm120-sglang-vs-vllm.md)(四堵墙调查)、[`fp8-blockscaled-backends.md`](fp8-blockscaled-backends.md)(FP8 后端调查)。
- **门控**:`vllm/utils/deep_gemm.py:is_deep_gemm_supported` = `VLLM_USE_DEEP_GEMM and has_deep_gemm() and support_deep_gemm()`;sm120 上 `support_deep_gemm()=False`。
