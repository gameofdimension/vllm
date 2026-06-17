#!/usr/bin/env bash
# revert.sh — 撤销 sm_120 补丁，把 vLLM wheel 恢复成干净状态。
#   1) 用 sm120/backups/*.orig 还原 6 个被编辑的文件；
#   2) 删除 4 个新增文件。
# 与 apply.sh 配对（apply.sh 的逆操作）。
set -euo pipefail
cd "$(dirname "$0")/.."

VLLM="$(uv run python - <<'PY'
import vllm, os
print(os.path.dirname(vllm.__file__))
PY
)"
[ -d "$VLLM" ] || { echo "ERROR: 找不到 vllm 包目录 ($VLLM)" >&2; exit 1; }

# 1. 还原 6 个被编辑文件（backups/<name>.orig -> wheel 路径）
restore() {  # <backups-filename> <wheel-relative>
  local src="sm120/backups/$1" dst="$VLLM/$2"
  if [ ! -f "$src" ]; then echo "WARN: 缺备份 $src，跳过"; return; fi
  cp "$src" "$dst"
  echo "RESTORED: $2  (from backups/$1)"
}
restore o_proj.py.orig              models/deepseek_v4/nvidia/ops/o_proj.py
restore flashmla.py.orig            models/deepseek_v4/nvidia/flashmla.py
restore flashinfer_sparse.py.orig   models/deepseek_v4/nvidia/flashinfer_sparse.py
restore fp8.py.orig                 model_executor/layers/quantization/fp8.py
restore indexer.py.orig             v1/attention/backends/mla/indexer.py
restore sparse_attn_indexer.py.orig model_executor/layers/sparse_attn_indexer.py

# 2. 删除 4 个新增文件
remove() {  # <wheel-relative>
  local dst="$VLLM/$1"
  if [ -f "$dst" ]; then rm -f "$dst"; echo "REMOVED:  $1"; else echo "SKIP (不存在): $1"; fi
}
remove models/deepseek_v4/nvidia/ops/sm120_o_proj.py
remove models/deepseek_v4/nvidia/flash_mla_sm120_triton.py
remove models/deepseek_v4/nvidia/flash_mla_sparse_fwd_sm120.py
remove model_executor/layers/fp8_paged_mqa_logits_sm120.py

echo
echo "✅ 已还原为干净 vLLM。重新打补丁: bash sm120/apply.sh"
