#!/usr/bin/env bash
set -xeuo pipefail

# path="/warehouse/DeepSeek-V4-Flash-FP8"
path="/warehouse/DeepSeek-V4-Flash"  # fp4 checkpoint — 目标：对齐 SGLang sm_120 配方

# eager 模式开关（默认关 = 生产：full CUDA graph，启动慢、推理快）：
#   bash launch.sh              生产（无 --enforce-eager）
#   bash launch.sh --eager      eager（跳过 CUDA graph 捕获，启动快、推理慢，适合调试）
#   bash launch.sh --no-eager   显式关闭（覆盖下面的 EAGER=1）
#   EAGER=1 bash launch.sh      用环境变量开启
eager_args=()
if [ "${EAGER:-0}" = "1" ]; then
  eager_args=(--enforce-eager)
fi
for arg in "$@"; do
  case "$arg" in
    --eager|-e|eager) eager_args=(--enforce-eager) ;;
    --no-eager)       eager_args=() ;;
    *) echo "launch.sh: 忽略未知参数 '$arg'（可用：--eager / --no-eager）" >&2 ;;
  esac
done

uv run vllm serve "${path}" \
  --served-model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.9 \
  "${eager_args[@]}"
