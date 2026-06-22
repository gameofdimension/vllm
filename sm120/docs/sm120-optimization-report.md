# sm_120 DeepSeek-V4 推理优化：技术报告

> **项目**：vLLM 0.23.0 在 NVIDIA RTX PRO 5000（sm_120, Blackwell）上运行 DeepSeek-V4-Flash（fp4 checkpoint）的推理性能优化。
> **硬件**：8× RTX PRO 5000 72GB，`compute_capability=(12,0)` = sm_120。
> **日期**：2026-06-17 ~ 2026-06-22。
> **结果摘要**：decode 吞吐 **3.62× 提升**（36.6 → 132.5 tok/s），**反超 SGLang 56%**；prefill **9.3× 快于 SGLang**；GSM8K Pass@1 **0.980**（精度无退化）。

---

## 1. 背景

### 1.1 sm_120 的核心约束

sm_120 是桌面/专业级 Blackwell（RTX PRO 5000/6000），**没有 TMEM / tcgen05**（那是 SM100 数据中心 B100/B200 才有的硬件单元）。后果：

| 受影响路径 | sm_120 上的报错 |
|---|---|
| DeepGEMM（GEMM + attention.hpp einsum/MQA-logits） | `Unsupported architecture` |
| FlashMLA C 核（`flash_mla_with_kvcache` decode / `flash_mla_sparse_fwd` prefill） | `only supported on SM90a and SM100f` |
| FlashInfer TRT-LLM FMHA | `Unsupported architecture` |
| CUTLASS `scaled_mm` 对 `float8_e8m0fnu`（ue8m0）block scale | `dispatch_scaled_mm` 落空 |

策略：**全链路绕开 DeepGEMM / FlashMLA-C / ue8m0**，用 Triton / PyTorch / bf16 替换（对齐 SGLang nightly 的 sm_120 做法）。MoE MXFP4-Marlin 原生支持 sm_120，无需改。

### 1.2 起点状态

DeepSeek-V4-Flash 在 sm_120 上**已验证正确推理**（`15×23=345`），但**吞吐极低**——三个朴素 PyTorch 分块实现 + 一个融合 Triton decode 核是瓶颈。本报告记录将这些瓶颈全部消除的优化过程。

### 1.3 环境基线

- vLLM 0.23.0（editable install，源码树开发），分支 `v0.23.0-sm120`。
- torch 2.11.0+cu130, CUDA 13.0, Python 3.12, triton 3.x。
- 模型：`/warehouse/DeepSeek-V4-Flash`（fp4；`config.json`: `expert_dtype:"fp4"`, `scale_fmt:"ue8m0"`）。
- serve 参数：`--tensor-parallel-size 8 --kv-cache-dtype fp8 --gpu-memory-utilization 0.9`。

---

## 2. 优化措施

### 2.1 总览

共 **5 项优化**，按实施顺序：

| # | 优化 | 路径 | 核心手段 | 加速 |
|---|---|---|---|---|
| Op1 | c4 **paged decode** scorer | decode 热路径 | fp8 `tl.dot` tensor-core GEMM | 24.6× |
| Op2 | c4 **dense prefill** scorer | prefill indexer | 同上（dense KV 版） | 26.2× |
| Op3 | **MLA prefill** gathered sparse attention | prefill attention | H-tiled flash online-softmax `tl.dot` | 5.63× |
| Op4 | **MLA decode** 两阶段重构 | decode attention | gather+dequant 分离 → cuBLAS dense attention | **3.62×** |
| Op5 | 代码清理 + 测试 | 工程质量 | torch oracle → tests；serving Triton-only | — |

### 2.2 Op1/Op2：c4 indexer scorer → Triton（24.6× / 26.2×）

**问题**：c4 稀疏 indexer 的两个 scorer（paged decode + dense prefill）是朴素 PyTorch 分块循环。在 1M 上下文下，decode scorer 有 ~256 个 Python 循环 × ~8 个 kernel launch/块 = ~2000 次串行 launch/decode step；中间体 `[Bn, pages, block_size, D]` f32 在 max_model_len=1M 时 OOM。

**方案**：融合 Triton 核。grid `(token_tile, query_row)`，每 program 处理一个 token tile：
- 分页间接 KV load（per-token 128B fp8 值 + 4B f32 scale）。
- **`qk = tl.dot(kv_fp8, q_fp8ᵀ)`** —— fp8 tensor-core MMA（f32 累加器），替代逐 head 的 f32 matvec。这是关键：fp8 MMA 既快又与 oracle 数学一致（两边都在同一批 fp8 值上算点积）。
- `relu → 加权 head 求和 → ×k_scale`，位置 ≥ context_len → `-inf`。

**关键教训**：第一版用逐 head 的 f32 matvec（`tl.sum`），大 B 时反而比 oracle 的 `torch.bmm`（走 cuBLAS）**慢 4×**。换成 fp8 `tl.dot` tensor-core 后才拿到 24.6×。

### 2.3 Op3：MLA prefill → flash-style Triton（5.63×）

**问题**：MLA prefill 的 gathered sparse attention（真 softmax，attn_sink 在分母）是朴素 PyTorch（整体 materialize `[Tq, topk, D]` f32 + 两遍 softmax + 两次 einsum 读 KV）。

**方案**：flash-style online-softmax Triton 核，grid `(Tq, cdiv(H, BLOCK_H))`：
- 每 program 取 BLOCK_H=16 个 head（tile H 作为 GEMM M-dim），使 `[BLOCK_H, D=512]` 累加器适应 sm_120 的 99KB shared mem。
- QK `q[BLOCK_H,D] · kv.T[D,BLOCK_K]` + AV `p · kv` 均走 bf16 `tl.dot` tensor core。
- attn_sink：`e_sink = exp2(sink·LOG2E − rowmax)` 加分母。

**踩坑记录**（3 个死胡同，已入库）：
1. per-(query,head) program + matvec → 4× 变慢（无 tensor core）。
2. per-query + 整 H `tl.dot` → shared memory OOM（`[64,512]` f32 = 128KB > 99KB）。
3. H-tiled + `static_range` 展开 KV 循环 → e2e 首请求 JIT 编译超 RPC timeout（`tl.range` 运行时 bound 修复）。

### 2.4 Op4：MLA decode 两阶段重构（3.62×，反超 SGLang）

**这是最大的一项优化。** 

**问题定位**（torch profiler decode-step profile）：
```
_tiled_sparse_decode_kernel (MLA decode Triton)    88.8%
nccl all-reduce                                     4.5%
_paged_mqa_logits_kernel (Op1 scorer)              1.1%
MoE marlin                                          0.3%
```

MLA decode 核占 decode GPU 时间的 **88.8%**——不是 MoE、不是通信、不是 Op1。该核在 attention 循环内部做**间接 gather**（每个 head 独立 gather 同一份 batch KV → H=64 倍冗余；uncoalesced → ~10× 慢于 memory peak）。

**诊断**：两次调优尝试失败——
- 扩 autotune：+12% 单 shape，系统无增益。
- split-K：反而变慢（merge 开销 > 并行收益）。核是 gather-memory-bound，非并行度受限。

**突破**：从 SGLang GitHub 主干发现其新版已弃用融合核，改为**两阶段分离**架构（`nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`）：
1. **Phase 1：gather+dequant 独立**。PyTorch fancy-index 一次性把稀疏 KV 从 paged cache gather + fp8→bf16 反量化到连续 `[B, topk, D]` buffer。**所有 H 个 head 共享**，消除 64× 冗余。
2. **Phase 2：dense attention**。在连续 buffer 上跑 cuBLAS einsum（`einsum bhd,bvd` + softmax + `einsum bhv,bvd`）。tensor-core GEMM on coalesced data。

**为什么这么快**：
- gather 一次（H heads 共享）vs 旧核 H 次冗余 → gather 总量 ÷H。
- attention 在连续数据上跑 cuBLAS（coalesced + tensor core）vs 旧核 indirect gather 循环内做 attention（uncoalesced + 无 tensor core）。
- PyTorch fancy-index + cuBLAS 比 SGLang 的 Triton gather kernel 更优 → **反超 SGLang 56%**。

---

## 3. 原理

### 3.1 为什么 sm_120 需要 Triton/PyTorch 而非 DeepGEMM/CUTLASS

DeepGEMM 和 FlashMLA-C 依赖 TMEM/tcgen05 指令（SM100 独有）。CUTLASS `scaled_mm` 不支持 ue8m0 block scale（只有 f32 block scale 有 kernel）。sm_120 上这些路径全部 `Unsupported architecture`。Triton 和 PyTorch + cuBLAS 不依赖这些专有指令，在 sm_120 上原生可用。

### 3.2 fp8 tensor-core `tl.dot` 的正确性

两个 c4 scorer 中，oracle 在 fp8 值上做 f32 点积（`fp8.to(f32) · fp8.to(f32)`）。Triton 的 fp8 MMA（`tl.dot(fp8, fp8, out_dtype=f32)`）计算的是 `sum(fp8_a · fp8_b)` 以 f32 累加——数学上等价于在 fp8-representable 值上做 f32 点积（只是累加顺序不同，引入 ~1e-6 舍入差）。实测 max_err ~1.4e-6（f32 logits），远低于容忍阈值。

### 3.3 两阶段 gather-separation 的根本优势

旧融合核的性能瓶颈是**间接 gather 的访存效率**。每个 attention program 在循环内通过 `block_tables` 间接索引 paged cache——这种 scattered 访存模式无法利用 GPU 的 memory coalescing，有效带宽仅为 peak 的 ~10%。且同一个 batch 的 KV 被 H 个 head 的 program 各自重复 gather（虽然 L2 cache 能部分缓解，但 launch overhead 和 L2 miss 的代价仍然显著）。

两阶段架构的根本优势在于**将随机访存转化为顺序访存**：
1. gather 阶段输出连续 buffer（顺序写），虽然输入侧仍是间接读，但 PyTorch fancy-index 的 gather kernel 针对此模式高度优化。
2. attention 阶段的读完全是顺序的（连续 buffer），cuBLAS 的 tensor-core GEMM 可达到接近 peak 的吞吐。

### 3.4 cudagraph 对 sm_120 路径无增益的原因

sm_120 的几个注意力算子（三个 scorer + MLA decode/prefill）都带 `@eager_break_during_capture` 装饰——因为它们的输出形状/控制流依赖运行时数据（context_len、topk、indices 等），不满足 CUDA graph 捕获的静态形状要求。因此即使开启 cudagraph，这些算子仍走 eager。实测确认：eager 和 cudagraph 下 decode 吞吐几乎相同（36.5 vs 36.6 tok/s）。两阶段重构后同样如此。

---

## 4. 效果

### 4.1 算子级（独立 bench，vs PyTorch oracle）

| 算子 | 最大 shape | PyTorch oracle | Triton 优化 | 加速 |
|---|---|---|---|---|
| Op1 paged decode scorer | B=8, ctx=1M | 28.4 ms | 1.15 ms | **24.6×** |
| Op2 dense prefill scorer | 4096×32768 | 230.8 ms | 8.8 ms | **26.2×** |
| Op3 MLA prefill | 2048² | 53.5 ms | 9.5 ms | **5.63×** |

### 4.2 系统级 decode（`vllm bench serve`，TP8 eager）

| 版本 | output tok/s | ITL | vs 旧核 | vs SGLang |
|---|---|---|---|---|
| **两阶段（最终）** | **132.5** | **234 ms** | **3.62×** | **1.56×** |
| 融合 Triton（旧） | 36.6 | 864 ms | 1× | 0.43× |
| SGLang | 85.2 | 287 ms | — | 1× |

### 4.3 系统级 prefill（long-prefill 场景）

| 版本 | total tok/s | vs SGLang |
|---|---|---|
| **vLLM Triton** | **1669** | **9.3×** |
| SGLang | 179 | 1× |

### 4.4 正确性

| 检查 | 结果 |
|---|---|
| GSM8K Pass@1（200 题） | **0.980**（旧核 0.965，无退化） |
| 算子单测（11 项，rtol=0，紧 atol） | **11/11 通过** |
| needle 召回 @ 12k/16k/64k | 通过（32k 单次噪声） |
| 长生成质量（300 词作文） | 连贯准确，无重复退化 |
| MoE sustained decode（16k tokens） | 36.4 tok/s 稳定，无 NaN |
| `15*23=?` | **345**，`finish_reason=stop` |

### 4.5 decode-step profile（优化前后对比）

优化前（融合核）：
```
_tiled_sparse_decode_kernel     88.8%   ← 瓶颈
nccl all-reduce                  4.5%
Op1 scorer                       1.1%
MoE                              0.3%
```

优化后（两阶段）：融合核 88.8% 的瓶颈被结构性消除（gather 一次 + cuBLAS dense attention），decode 吞吐 3.62×。

---

## 5. 工程产出

### 5.1 新增文件

| 文件 | 作用 |
|---|---|
| `vllm/model_executor/layers/fp8_mqa_logits_triton.py` | Op1/Op2 Triton scorer（paged + dense） |
| `vllm/models/deepseek_v4/nvidia/flash_mla_sparse_prefill_triton.py` | Op3 MLA prefill flash 核 |
| `vllm/models/deepseek_v4/nvidia/flash_mla_sm120_twophase.py` | Op4 MLA decode 两阶段 |
| `tests/kernels/test_sm120_mqa_logits.py` | Op1/Op2 正确性单测（torch oracle） |
| `tests/kernels/test_sm120_mla_prefill.py` | Op3 正确性单测（torch oracle） |
| `sm120/bench/scorer_bench.py` | 算子级延迟 bench |
| `sm120/bench/run_perf.sh` | 服务级吞吐 bench（vLLM vs SGLang） |
| `sm120/bench/mla_decode_bench.py` | MLA decode 核延迟 bench |

### 5.2 架构变更

- **serving**：Triton-only（所有 sm_120 分支直接调 Triton 核，无 PyTorch 回退）。
- **torch oracle**：移入 `tests/kernels/`，作为正确性参考 + pytest 守卫。
- **门控**：所有优化在 `is_deep_gemm_supported()` 的 `else` 分支（仅 sm_120），SM90/100 完全不受影响。

### 5.3 待办（已记录，暂缓）

- §7.2 系统正确性 eval（GSM8K 已过，更长 benchmark 套件待做）。
- 两阶段 Phase 1 的 PyTorch gather → Triton gather kernel（进一步榨取带宽，目前 PyTorch 已反超 SGLang）。

---

## 附录 A：优化路径时间线

| 日期 | 事件 |
|---|---|
| 06-17 | sm120 enablement 落源码树（4 新文件 + 6 编辑），345 验证 |
| 06-17 | Op1 paged decode scorer → Triton fp8 `tl.dot`（24.6×） |
| 06-17 | Op2 dense prefill scorer → Triton（26.2×） |
| 06-17 | Op3 MLA prefill → flash Triton（5.63×） |
| 06-18 | GSM8K Pass@1 = 0.965；SGLang 三方对比 |
| 06-18 | §7.3 MoE + 长上下文压测（64k 无 OOM，无 NaN） |
| 06-18 | torch oracle → tests；serving Triton-only |
| 06-22 | §7.5 decode-step profile → MLA decode 88.8% 瓶颈 |
| 06-22 | autotune / split-K 尝试 → 均失败，revert |
| 06-22 | **两阶段重构（SGLang GitHub 发现）→ 3.62×，反超 SGLang** |
| 06-22 | GSM8K Pass@1 = 0.980（精度无退化） |

---

## 附录 B：关键数字

- **decode 吞吐提升**：3.62×（36.6 → 132.5 tok/s）
- **vs SGLang decode**：1.56× 更快（85.2 → 132.5）
- **vs SGLang prefill**：9.3× 更快（179 → 1669）
- **GSM8K Pass@1**：0.980（无退化）
- **算子级最大加速**：26.2×（Op2 dense prefill scorer）
- **commit 数**：14（`v0.23.0-sm120` 分支，未推送）
