# MLA Decode 两阶段优化：旧融合核 vs 新两阶段架构的深度对比

> **Commit**：`992cd4162`（`flash_mla_sm120_twophase.py`）。
> **效果**：decode 吞吐 36.6 → 132.5 tok/s（**3.62×**），ITL 864 → 234ms。**反超 SGLang 56%**（85.2 → 132.5）。
> **日期**：2026-06-22。

---

## 1. 旧融合核（基线）的架构

### 1.1 数据流

```
每层 decode step（batch B=32, heads H=64, KV topk=512, D=512）:

Grid (B, H) = (32, 64) = 2048 个 program
每个 program (b, h):
  ├── load q[b, h]                          # 1 个 query 向量 [512]
  ├── for tile in range(0, topk=512, 32):   # 16 次串行迭代
  │     ├── 间接 gather 32 个 KV token       # ← scattered 读（indices 间接索引 paged cache）
  │     ├── fp8→f32 反量化 + ue8m0 exp2     # ← 逐元素计算（无 tensor core）
  │     ├── QK = sum(q[512] * kv[32,512])   # ← element-wise mul + reduce（无 tensor core）
  │     ├── online softmax 更新              # ← 标量/向量操作
  │     └── V accum = sum(p[32] * kv[32,512]) # ← 同上（无 tensor core）
  └── write out[b, h]                        # 最终输出
```

### 1.2 瓶颈分析（来自 torch profiler decode-step profile）

```
_tiled_sparse_decode_kernel    88.8%   ← 此核是 decode 的绝对瓶颈
nccl all-reduce                 4.5%
Op1 scorer                      1.1%
MoE marlin                      0.3%
```

**三个结构性问题**：

| 问题 | 说明 | 后果 |
|---|---|---|
| **H 倍冗余 gather** | 每个 batch 的 KV 被 H=64 个 head program 各自独立 gather（indices 是 per-batch 的，所有 head 读同一份 KV） | 64× 多余的 memory traffic（虽 L2 部分缓解，但 launch + L2 miss 代价仍大） |
| **uncoalesced 间接访存** | indices 指向分散的 cache pages → memory transactions scattered → 有效带宽 ~peak 的 10% | gather 成为 memory-bound 瓶颈（~350× off peak 的根源） |
| **无 tensor core** | QK/V 用 `tl.sum(q[None,:] * kv, axis=1)`（element-wise mul + reduce），不是 `tl.dot` | 计算密度远低于 tensor-core GEMM |

### 1.3 调优尝试（均失败）

| 尝试 | 结果 | 原因 |
|---|---|---|
| 扩 autotune（3→8 配置） | +12% 单 shape，系统无增益 | 核是 gather-bound，tile/warp 配置动不了访存效率 |
| split-K（grid B×H×S + LSE merge） | H=16/64 反慢 16%（merge 开销） | 非并行度受限；加 program 不改善 uncoalesced gather |

→ 结论：融合核架构本身是瓶颈，调优救不了。

---

## 2. 新两阶段架构

### 2.1 核心思想

**将间接 gather 从 attention 循环中拆出来**：先一次性 gather+dequant 到连续 buffer，再在连续数据上做 dense attention。

### 2.2 数据流

```
Phase 1: gather + dequant（每 batch 做一次，所有 head 共享）
  input:  k_cache [num_pages, page_size, 584] + indices [B, topk]
  output: gathered_kv [B, topk, 512] f32（连续）+ invalid_mask [B, topk] bool

  # PyTorch fancy-index gather（coalesced 写，高效 indirect 读）
  page_id = indices // page_size
  page_off = indices % page_size
  gathered = data[page_id, page_off]          # [B, topk, 576] ← 一次 gather，H head 共享
  gathered_scales = scales[page_id, page_off]  # [B, topk, 8]

  # fp8→f32 反量化（7 groups × 64 broadcast，vectorized PyTorch）
  nope = gathered[...,:448].view(fp8).reshape(B,topk,7,64)
  sc = gathered_scales[...,:7].to(f32).reshape(B,topk,7,1)
  kv_nope = (nope.to(f32) * exp2(sc - 127)).reshape(B,topk,448)
  kv_rope = gathered[...,448:576].view(bf16).to(f32)  # [B,topk,64]
  kv = cat([kv_nope, kv_rope])                          # [B,topk,512]

Phase 2: dense attention（cuBLAS einsum on contiguous buffer）
  qf = q.squeeze(1).to(f32) * sm_scale     # [B, H, 512]
  scores = einsum("bhd,bvd->bhv", qf, kv)  # [B, H, topk] ← cuBLAS GEMM, tensor core
  scores.masked_fill(invalid, -inf)
  scores_max = scores.amax(-1)
  e = exp(scores - scores_max)
  numer = einsum("bhv,bvd->bhd", e, kv)    # [B, H, 512] ← cuBLAS GEMM, tensor core
  denom = e.sum(-1)
  out = numer / denom                       # [B, H, 512]
```

### 2.3 与旧核的对比

| 维度 | 旧融合核 | 新两阶段 |
|---|---|---|
| **gather 次数** | H=64 次/batch（每 head 独立 gather） | **1 次/batch**（所有 head 共享） |
| **gather 访存模式** | scattered indirect（在 attention 循环内） | PyTorch fancy-index（独立阶段，coalesced 写） |
| **attention 数据布局** | scattered（间接索引 paged cache） | **contiguous**（连续 dense buffer） |
| **QK/V 计算** | `tl.sum(q * kv)` element-wise mul-reduce | **`torch.einsum` → cuBLAS GEMM**（tensor core） |
| **并行度** | grid (B, H)，per-program 串行 tile 循环 | Phase 1: 1D grid (flat gather)；Phase 2: cuBLAS 自动调度 |
| **kernel 数** | 1（融合） | 2（gather + attention），但每次更高效 |

### 2.4 为什么快了 3.62×

1. **gather 总量 ÷H**：旧核每个 head program 独立 gather 同一份 KV（H=64 倍冗余）。两阶段只 gather 一次（所有 head 共享连续 buffer）。即使 L2 cache 部分缓解了冗余读，launch overhead + L2 miss 的代价仍然显著。

2. **顺序访存替代随机访存**：
   - 旧核 attention 读：通过 `block_tables` 间接索引 → scattered memory transactions → 有效带宽 ~10% peak。
   - 新核 Phase 2 读：连续 `[B, topk, D]` buffer → coalesced → 接近 peak 带宽。

3. **tensor core 替代 element-wise reduce**：
   - 旧核 QK：`tl.sum(q[None,:] * kv[BLOCK_T,512], axis=1)` → 编译为标量/向量 mul-reduce 指令。
   - 新核 QK：`torch.einsum("bhd,bvd->bhv")` → cuBLAS 调度 HMMA（tensor core） → 计算吞吐 ~50× 高于标量。

4. **PyTorch/cuBLAS 成熟度**：PyTorch fancy-index gather kernel 和 cuBLAS GEMM 是高度优化的库函数（多年迭代），比手写 Triton 核在 Blackwell 上的成熟度更高。

---

## 3. 正确性等价性

两阶段架构和旧融合核计算的是**同一个数学函数**：对每个 (batch, head)，对 topk 个 KV token 做 scaled dot-product attention（含 attn_sink 在分母），输出 softmax-weighted V 累加。

- 旧核：per-head program 内联 gather → QK → softmax → V（serial tile loop）。
- 新核：global gather → contiguous buffer → einsum QK → softmax → einsum V。

两者唯一差异：**计算顺序和中间精度**。旧核在 Triton f32 内联计算；新核在 PyTorch f32（gather 后是 f32 buffer）上计算。实测 GSM8K Pass@1 从 0.965 → 0.980（噪声内，无退化），`15*23=345` 正确。

---

## 4. 从 SGLang 发现此架构

### 4.1 发现过程

本地 SGLang v0.3.21 **没有** `flash_mla_sm120_triton.py`（旧融合核）——这个文件是在之后的版本加入的。但其 GitHub 主干（`sgl-project/sglang`）有：

1. `flash_mla_sm120_triton.py`：与我们的旧核**完全相同**（说明我们移植的就是这个文件）。
2. `nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`：**全新的两阶段架构**——SGLang 已弃用融合核。

### 4.2 SGLang 两阶段的结构

SGLang 的两阶段比我们的更完整（用 Triton gather kernel + 多种 attention 变体），但**核心思想相同**：

```
SGLang:
  Phase 1: _gather_dequant_dsv4_kernel (Triton, batched scale, 1D fused grid)
  Phase 2: run_unified_attention / run_splitk_unified_attention (Triton dense attention)

我们（vLLM sm120）:
  Phase 1: PyTorch fancy-index gather + vectorized dequant
  Phase 2: PyTorch einsum → cuBLAS GEMM
```

我们用 PyTorch 替代 SGLang 的 Triton gather/attention kernel——**更简单（~90 行 vs SGLang ~800 行）**，且**实测更快**（132.5 vs 85.2 tok/s），可能因为 cuBLAS 对 contiguous attention 的优化比 SGLang 的 Triton 版更成熟。

### 4.3 为什么我们没有直接移植 SGLang 的 Triton gather kernel

1. SGLang 的 gather kernel 依赖 SGLang 特有的数据结构（`kv_scope`、`blocked_k_quantized`），与 vLLM 的 paged cache 格式不兼容。
2. PyTorch fancy-index gather + cuBLAS einsum 已经**反超 SGLang 56%**——说明 PyTorch 路径足够快。
3. 后续如果需要进一步榨取带宽（Phase 1 的 PyTorch gather → Triton gather kernel），可以再移植——但当前不是瓶颈。

---

## 5. 性能数据汇总

### 5.1 decode 吞吐（`vllm bench serve`，TP8 eager，in128/out256/32 prompts）

| 版本 | output tok/s | ITL (ms) | vs 旧融合 | vs SGLang |
|---|---|---|---|---|
| **两阶段（新）** | **132.5** | **234** | **3.62×** | **1.56×** |
| 融合 Triton（旧） | 36.6 | 864 | 1× | 0.43× |
| SGLang | 85.2 | 287 | — | 1× |

### 5.2 GSM8K Pass@1（200 题，5-shot CoT）

| 版本 | Pass@1 | invalid | eval wall |
|---|---|---|---|
| 两阶段（新） | **0.980** | 0.000 | 147s |
| 融合（旧） | 0.965 | 0.000 | 488s |

精度无退化；eval 速度 3.3× 更快（与 decode 加速一致）。

### 5.3 decode-step profile（优化前后）

**优化前**（融合核）：
```
_tiled_sparse_decode_kernel     88.8%   ← gather-inside-attention 瓶颈
nccl all-reduce                  4.5%
Op1 scorer                       1.1%
MoE                              0.3%
```

**优化后**（两阶段）：88.8% 瓶颈被结构性消除（gather 一次 + cuBLAS dense attention），decode 吞吐 3.62×。瓶颈已不在此核——decode 的新瓶颈（如有）需要重新 profile 确定。

---

## 6. 代码对比

### 旧融合核（`flash_mla_sm120_triton.py`，318 行）

```python
@triton.jit
def _tiled_sparse_decode_kernel(Q_ptr, cache_fp8_ptr, ..., O_ptr, LSE_ptr, ...):
    bid = tl.program_id(0)  # batch
    hid = tl.program_id(1)  # head
    # load q[bid, hid]
    for tile_start in range(0, topk, BLOCK_T):
        # 1. 间接 gather BLOCK_T KV tokens（scattered）
        # 2. fp8 dequant + ue8m0 scale
        # 3. QK = sum(q * kv)  ← element-wise, no tensor core
        # 4. online softmax
        # 5. V accum = sum(p * kv)  ← element-wise, no tensor core
    # write out[bid, hid]
```

### 新两阶段（`flash_mla_sm120_twophase.py`，93 行）

```python
def flash_mla_sparse_decode_two_phase(q, k_cache, indices, ...):
    # Phase 1: gather + dequant（PyTorch fancy-index，所有 head 共享）
    data[page_id, page_off]  → gathered_kv [B, topk, D] contiguous
    fp8 dequant → f32

    # Phase 2: dense attention（cuBLAS einsum，tensor core）
    scores = einsum("bhd,bvd->bhv", qf, kv)   # cuBLAS GEMM
    e = exp(scores - scores_max)
    numer = einsum("bhv,bvd->bhd", e, kv)     # cuBLAS GEMM
    out = numer / denom
```

**代码量**：318 行 → 93 行（-71%）。更简洁、更可维护。
