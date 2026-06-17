# vLLM FP8 块缩放 GEMM 的三种 Backend

> 适用场景：用 vLLM 0.23.0 在 **NVIDIA RTX PRO 5000 (sm_120)** 上部署 **DeepSeek-V4-Flash**（权重为 FP8 e4m3 + `[128,128]` 块缩放，`scale_fmt=ue8m0`）时的 kernel 选择与排障记录。
> 关联文件：`launch.sh`、模型 `config.json`、日志 `/tmp/vllm_launch.log`、`/tmp/vllm_triton2.log`。
> 配套文档：sm_120 上 SGLang vs vLLM 的完整对比与移植方案见 [`deepseek-v4-sm120-sglang-vs-vllm.md`](./deepseek-v4-sm120-sglang-vs-vllm.md)（本文是其中"FP8 线性层 backend"那堵墙的展开）。

---

## 1. 背景：这些 "backend" 到底是什么

当模型的线性层权重是 **FP8 + block scale**（块级缩放）量化时，vLLM 的 `Fp8LinearMethod` 不会直接调用普通 `torch.mm`，而是走一个**可插拔的 kernel 选择器** `choose_scaled_mm_linear_kernel`，从一组候选实现里挑一个来执行块缩放矩阵乘。

所谓 "三种 backend"，就是这个选择器里**三个最相关的候选实现**：

| Backend | 实现位置 | `apply_block_scaled_mm` 内部用什么算 |
|---|---|---|
| **CUTLASS** | `model_executor/kernels/linear/scaled_mm/cutlass.py` | PyTorch 预编译进 wheel 的 `torch.ops._C.cutlass_scaled_mm`（libtorch 自带 CUTLASS c3x，与硬件强绑定） |
| **Triton** | `model_executor/kernels/linear/scaled_mm/triton.py` | JIT 出来的 Triton kernel `w8a8_triton_block_scaled_mm`，哪里能跑 Triton 就能跑 |
| **DeepGEMM** | `model_executor/kernels/linear/scaled_mm/deep_gemm.py` | DeepSeek 自家 GEMM 库（vLLM 内 vendored 了一份），**唯一原生支持 `ue8m0`** |

三者都继承同一个抽象基类 `Fp8BlockScaledMMLinearKernel`（`BlockScaledMMLinearKernel.py`），只各自实现 `apply_block_scaled_mm(A, B, As, Bs)`：输入 FP8 的激活 `A` 与权重 `B`、各自的块缩放因子 `As` / `Bs`，输出反量化后的矩阵乘结果。

---

## 2. 完整候选列表与优先级

选择器从 `model_executor/kernels/linear/__init__.py:311` 的优先级列表自上而下遍历，取第一个 `is_supported()` 且 `can_implement()` 同时通过的 kernel。FP8 block 这条在 CUDA 上一共 **5** 个候选（不止 3 个）：

```python
# model_executor/kernels/linear/__init__.py:307-317
_POSSIBLE_FP8_BLOCK_KERNELS[CUDA] = [
    FlashInferFp8DeepGEMMDynamicBlockScaledKernel,  # ① 需 FlashInfer + DeepGEMM
    DeepGemmFp8BlockScaledMMKernel,                 # ② ue8m0 原生 —— 被 sm_120 架构 gate 挡掉
    CutlassFp8BlockScaledMMKernel,                  # ③ 默认落点（auto 选中的就是这个）
    MarlinFP8ScaledMMLinearKernel,                  # ④ 仅 per-tensor / per-channel，不适配 block
    TritonFp8BlockScaledMMKernel,                   # ⑤ 仅 --linear-backend triton 时被强制选
]
```

> 本文重点讲的「三种」即 ②DeepGEMM、③CUTLASS、⑤Triton。①同样依赖 DeepGEMM、会撞同样的架构问题；④对 block 配置 `can_implement` 一般为 False。

---

## 3. 选择机制

`choose_scaled_mm_linear_kernel`（`__init__.py:450`）按上面的优先级顺序逐个判断：

- `is_supported(compute_capability)`：硬件/平台层面是否支持（架构、CUDA、平台等）。
- `can_implement(config)`：该层配置（dtype、块大小、量化格式）是否可被实现。

**两种手动覆盖：**

| 方式 | 作用 | 代码 |
|---|---|---|
| `--linear-backend <name>` | 把候选集合**过滤**成某一组 backend 的 kernel（auto 时不过滤） | `__init__.py:194` `_LINEAR_BACKEND_KERNEL_MAP`；argparse 在 `engine/arg_utils.py:1452` |
| `VLLM_DISABLED_KERNELS=类A,类B` | **点名禁用**某些 kernel 类 | `envs.py:111`，过滤逻辑在 `__init__.py:428` |

`--linear-backend` 名字到 kernel 类集合的映射（节选）：

```python
# __init__.py:194-257
"cutlass":   { CutlassFp8BlockScaledMMKernel, ... }
"triton":    { TritonFp8BlockScaledMMKernel, ... }
"deep_gemm": { DeepGemmFp8BlockScaledMMKernel }
```

---

## 4. 三种 Backend 详解

### 4.1 DeepGEMM（理想解，但本机被挡）

- **为什么理想**：原生支持 `ue8m0`（MXFP8 微指数块缩放）格式。vLLM 启动时甚至会打印 `Detected quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM`。
- **启用条件**（`deep_gemm.py:47-52` → `utils/deep_gemm.py:96`）：

```python
is_deep_gemm_supported() =
      envs.VLLM_USE_DEEP_GEMM          # 默认 True
  and has_deep_gemm()                  # 外部包优先，回退 vLLM 自带 vendored 副本
  and current_platform.support_deep_gemm()   # ← 本机 False
```

- **架构 gate**（`platforms/cuda.py:557`）：

```python
def support_deep_gemm(cls):
    return cls.is_device_capability(90) or cls.is_device_capability_family(100)
    # is_device_capability_family (platforms/interface.py:363):
    #   (cap // 10) == (capability // 10)
```

  只覆盖 **Hopper (sm_90)** 与 **Blackwell 数据中心 family-100 (sm_10x，B100/B200)**。
  本机 RTX PRO 5000 是 **sm_120 (family 12.x)**，`120//10=12 ≠ 10` → `support_deep_gemm()=False` → 被跳过。

- **关键结论**：`has_deep_gemm()` 在本机是 **True**（vendored 副本可 import）。装外部 `deep_gemm==1.0.0` 没用——它只影响 `has_deep_gemm()`（已 True），**碰不到架构 gate**。

### 4.2 CUTLASS（默认 auto 命中点）

- **为什么被选**：`is_supported()` 对任意 CUDA 架构都返回 True，而排在它前面的 DeepGEMM 已被架构 gate 筛掉，于是它成为第一个 `is_supported && can_implement` 双通过的候选 → 日志 `Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod`。
- **调用链**：`cutlass.py:324 apply_block_scaled_mm` → `ops.cutlass_scaled_mm` → `torch.ops._C.cutlass_scaled_mm`（PyTorch 自带 CUTLASS）。
- **为什么崩**：CUTLASS c3x 没有为 **e8m0 块缩放 FP8** 注册 kernel，dispatch 遍历后落空：

```
RuntimeError: dispatch_scaled_mm,
/workspace/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp:17
```

### 4.3 Triton（强制 `--linear-backend triton` 才会选）

- **为什么默认走不到**：它排在优先级列表第 5 位，CUTLASS 先被选中就不再往下看。只有用 `--linear-backend triton` 把候选集合过滤成 triton 组，才会选 `TritonFp8BlockScaledMMKernel`。
- **调用链**：`triton.py:173 apply_block_scaled_mm` → `torch.ops.vllm.w8a8_triton_block_scaled_mm_func` → `triton.py:198` → `fp8_utils.py:906 w8a8_triton_block_scaled_mm`。
- **为什么崩**：Triton 自己在 JIT 绑定参数做 dtype 规范化时，字典里没有 `float8_e8m0fnu` 这个键——**还没进 kernel 就 KeyError**：

```
File ".../triton/_utils.py", line 105, in canonicalize_dtype
    return type_canonicalisation_dict[dtype_str]
KeyError: 'float8_e8m0fnu'
```

---

## 5. 本机故障链小结

崩溃发生在初始化 warmup/profiling 前向，位置在 DeepSeek-V4 attention 的融合 WQA/WKV GEMM（`vllm/models/deepseek_v4/attention.py:411 fused_wqa_wkv`），8 个 TP worker 一致崩溃，复用同一被选中的 block kernel。

**核心枢纽**：唯一能处理 `ue8m0` 的 DeepGEMM，因 `support_deep_gemm()` 不覆盖 sm_120 而早退 → 选择器回退到 CUTLASS/Triton → 两者都死在 `float8_e8m0fnu`。

| Backend | 默认能否选中 | 本机结果 |
|---|---|---|
| DeepGEMM | 否（架构 gate 挡掉） | `is_supported()=False`，跳过 |
| CUTLASS | **是（auto 命中）** | `dispatch_scaled_mm … helper:17`（无 e8m0 kernel） |
| Triton | 否（优先级靠后） | `--linear-backend triton` 强选后 `KeyError: 'float8_e8m0fnu'` |

---

## 6. 如何探测 / 调试

```bash
# ① 看日志里选了哪个（auto 模式）
grep -nE "Selected .*LinearMethod" /tmp/vllm_launch.log

# ② 探测 deep_gemm 是否可用 + 能否启用（这才是 vLLM 的真实 gate）
uv run python -c "
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.platforms import current_platform as p
print('has_deep_gemm()          =', has_deep_gemm())          # 能否 import（外部包优先，回退 vendored）
print('is_deep_gemm_supported() =', is_deep_gemm_supported()) # 真正决定用不用
print('cap                      =', p.get_device_capability())
print('support_deep_gemm()      =', p.support_deep_gemm())
"

# ③ 强制某 backend 跑一次看报错
#    --linear-backend {cutlass|triton|deep_gemm|flashinfer_cutlass|marlin|...}
```

> 注意：裸 `import deep_gemm` 和 `uv pip show deep_gemm` 都查不到 vLLM 自带的 vendored 副本，**别用它们判断**——要以 `has_deep_gemm()` 为准。

---

## 7. 可行的修复方向

1. **强行打开架构 gate**：启动前 monkeypatch `CudaPlatform.support_deep_gemm`（或 `is_device_capability_family`）对 sm_120 返回 True。风险——DeepGEMM 自身 kernel 得真为 sm_120 编了，否则换种姿势崩（未实测）。
2. **换权重**：检查注释掉的 `/warehouse/DeepSeek-V4-Flash-FP8`，若其 `scale_fmt` 是 `float32`（非 `ue8m0`），CUTLASS 可 dispatch，无需 DeepGEMM。
3. **升级**：换一版把 sm_120 写进 `support_deep_gemm()`、且 deep_gemm 带 sm_120 kernel 的 vLLM。

`--enforce-eager` 对本问题无效——报错在算子级别，与 cudagraph 捕获无关。
