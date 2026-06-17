# sm120/ — DeepSeek-V4 on RTX PRO (sm_120) enablement for vLLM

本目录承载"让 vLLM 0.23.0 在 **NVIDIA RTX PRO 5000/6000（Blackwell, sm_120）** 上跑起 DeepSeek-V4-Flash（fp4 checkpoint）"的改动与记录。背景与完整方案见 [`docs/deepseek-v4-sm120-sglang-vs-vllm.md`](docs/deepseek-v4-sm120-sglang-vs-vllm.md)。

**状态：✅ 已验证可用**——服务器正常起服务，`chat/completions` "Say hello"→`"Hello!"`、"capital of France is"→`" Paris."`。剩余仅速度优化（朴素 PyTorch scorer 慢）。

## 核心思路
sm_120 没有 DeepGEMM/FlashMLA 所需的 TMEM/tcgen05（SM100 数据中心 Blackwell 才有），所以凡硬依赖它们的路径都换成 sm_120 能跑的实现（参考 SGLang nightly）：o_proj 退回 bf16、MLA decode 用移植的 Triton 核、MLA prefill + 2 个 indexer scorer 用 PyTorch、FP8 线性层 ue8m0 scale 转 f32。MoE MXFP4-Marlin 原生支持，无需改。

## 目录结构
```
sm120/
├── README.md            本文件
├── CHANGES.md           ★ wheel 改动清单（每条改动 + 备份引用 + 验证记录）
├── apply.sh             ★ 一键重打补丁（vLLM 被重装后用）
├── revert.sh            ★ 一键还原（恢复 .orig + 删新文件）
├── backups/             被编辑文件的原始 .orig 副本（6 个）
├── patched/             6 个已打补丁文件的 canonical 版（apply.sh 复制源）
└── 4 个新模块的 canonical 源（apply.sh 复制源）：
    ├── sm120_o_proj.py            （o_proj bf16 实现）
    ├── flash_mla_sm120_triton.py  （MLA decode Triton 核，移植自 SGLang）
    ├── flash_mla_sparse_fwd_sm120.py （MLA prefill，PyTorch）
    └── fp8_paged_mqa_logits_sm120.py  （2 个 FP8 indexer scorer，PyTorch）
```

## 操作
```bash
# vLLM 被 uv sync / pip 重装后，补丁会丢 —— 一键重打：
bash sm120/apply.sh

# 撤销所有补丁，恢复干净 vLLM：
bash sm120/revert.sh

# 起服务验证：
bash launch.sh        # 指向 fp4 checkpoint，无需 config override
```

## 改动概览（权威清单见 CHANGES.md）
所有改动针对已安装 wheel `.venv/.../vllm/`，均门控 `is_deep_gemm_supported()`，**SM90/SM100 行为不变**，只在 sm_120（`get_device_capability()==(12,0)`）走新路径。

- **新文件（4）**：`sm120_o_proj.py`、`flash_mla_sm120_triton.py`、`flash_mla_sparse_fwd_sm120.py`、`fp8_paged_mqa_logits_sm120.py`
- **编辑文件（6）**：`ops/o_proj.py`、`nvidia/flashmla.py`、`nvidia/flashinfer_sparse.py`、`quantization/fp8.py`、`mla/indexer.py`、`sparse_attn_indexer.py`

## 注意
- 补丁是 **vLLM 0.23.0 版本绑定**的（patched/ 是该版本的整文件覆盖）。换 vLLM 版本需重新移植，不能直接 apply。
- apply.sh 会检测版本并在 ≠0.23.0 时警告。
