#!/usr/bin/env bash
# apply.sh — 把 sm_120 DeepSeek-V4 补丁重新打进已安装的 vLLM wheel。
#
# 用途：vLLM 被 `uv sync` / pip 重装后，wheel 里的补丁会丢失；本脚本从
# sm120/ 下的 canonical 副本（4 个新文件 + sm120/patched/ 下的 6 个已打补丁
# 版）一次性 cp 回去。与 revert.sh 配对。
#
# 全部改动门控 is_deep_gemm_supported()，SM90/SM100 行为不变，只在 sm_120 生效。
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. 定位已安装的 vllm 包目录
VLLM="$(uv run python - <<'PY'
import vllm, os
print(os.path.dirname(vllm.__file__))
PY
)"
[ -d "$VLLM" ] || { echo "ERROR: 找不到 vllm 包目录 ($VLLM)；先 uv add vllm==0.23.0" >&2; exit 1; }
VER="$(uv run python -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo unknown)"
echo "vllm $VER @ $VLLM"
if [ "$VER" != "0.23.0" ]; then
  echo "WARNING: 补丁针对 vLLM 0.23.0 编写，当前是 $VER —— 直接覆盖同路径文件，若版本不同可能需要重新移植。" >&2
fi

copy() {  # <src-relative-to-repo-root> <wheel-relative>
  local src="$1" dst="$VLLM/$2"
  [ -f "$src" ] || { echo "SKIP (无源文件): $src"; return; }
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  echo "APPLIED: $1 -> $2"
}

echo "--- 4 个新文件（canonical 在 sm120/） ---"
copy sm120/sm120_o_proj.py              models/deepseek_v4/nvidia/ops/sm120_o_proj.py
copy sm120/flash_mla_sm120_triton.py    models/deepseek_v4/nvidia/flash_mla_sm120_triton.py
copy sm120/flash_mla_sparse_fwd_sm120.py models/deepseek_v4/nvidia/flash_mla_sparse_fwd_sm120.py
copy sm120/fp8_paged_mqa_logits_sm120.py model_executor/layers/fp8_paged_mqa_logits_sm120.py

echo "--- 6 个已打补丁文件（canonical 在 sm120/patched/） ---"
copy sm120/patched/o_proj.py              models/deepseek_v4/nvidia/ops/o_proj.py
copy sm120/patched/flashmla.py            models/deepseek_v4/nvidia/flashmla.py
copy sm120/patched/flashinfer_sparse.py   models/deepseek_v4/nvidia/flashinfer_sparse.py
copy sm120/patched/fp8.py                 model_executor/layers/quantization/fp8.py
copy sm120/patched/indexer.py             v1/attention/backends/mla/indexer.py
copy sm120/patched/sparse_attn_indexer.py model_executor/layers/sparse_attn_indexer.py

echo "--- import 校验 ---"
if uv run python -c "
from vllm.models.deepseek_v4.nvidia import flashmla, flashinfer_sparse
from vllm.models.deepseek_v4.nvidia.ops.o_proj import sm120_o_proj
from vllm.models.deepseek_v4.nvidia.flash_mla_sm120_triton import flash_mla_sparse_decode_triton
from vllm.models.deepseek_v4.nvidia.flash_mla_sparse_fwd_sm120 import flash_mla_sparse_fwd_sm120
from vllm.model_executor.layers.fp8_paged_mqa_logits_sm120 import fp8_paged_mqa_logits_sm120
import vllm.model_executor.layers.sparse_attn_indexer, vllm.model_executor.layers.quantization.fp8
print('OK: 所有补丁模块可导入')
" 2>&1 | tail -3; then
  echo
  echo "✅ 补丁已应用。还原: bash sm120/revert.sh"
else
  echo "ERROR: import 校验失败，请检查上面输出" >&2
  exit 1
fi
