set -xeuo pipefail

# path="/warehouse/DeepSeek-V4-Flash-FP8"
path="/warehouse/DeepSeek-V4-Flash"  # fp4 checkpoint — 目标：对齐 SGLang sm_120 配方

uv run vllm serve "${path}" \
  --served-model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.9

#  --enforce-eager \

