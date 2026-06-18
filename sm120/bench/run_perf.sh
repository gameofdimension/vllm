#!/usr/bin/env bash
# sm_120 perf benchmark runner — Triton vs torch (our vLLM) vs SGLang.
#
# Provider-agnostic: hits an OpenAI-compatible /v1/completions endpoint, so the
# SAME scenarios run against our vLLM server (Triton or torch) AND an SGLang
# server. Synthetic random prompts of FIXED input/output length (reproducible,
# content-agnostic for throughput). Each request generates exactly output_len
# tokens (--ignore-eos).
#
# Two scenarios isolate where the sm_120 Triton ops help:
#   A) decode-heavy  (short in / long out) -> exercises Op1 (paged decode c4 scorer)
#   B) long-prefill (long in / short out) -> exercises Op2 (dense prefill scorer) + Op3 (MLA prefill)
#
# Usage:
#   run_perf.sh <label> <base_url> [model_name]
# e.g.
#   # vLLM Triton:   bash sm120/bench/run_perf.sh triton  http://localhost:8000
#   # vLLM torch:    bash sm120/bench/run_perf.sh torch   http://localhost:8000   # (server started with VLLM_SM120_TRITON_*=0)
#   # SGLang:        bash sm120/bench/run_perf.sh sglang  http://<sglang-host>:<port> deepseek-ai/DeepSeek-V4-Flash
set -euo pipefail

LABEL=${1:?usage: run_perf.sh <label> <base_url> [model]}
URL=${2:?usage: run_perf.sh <label> <base_url> [model]}
MODEL=${3:-deepseek-ai/DeepSeek-V4-Flash}
TOK=/warehouse/DeepSeek-V4-Flash
OUT=sm120/bench/results
mkdir -p "$OUT"

COMMON=(--backend openai --base-url "$URL" --endpoint /v1/completions
        --dataset-name random --ignore-eos --seed 42 --request-rate inf
        --tokenizer "$TOK" --tokenizer-mode deepseek_v4 --trust-remote-code
        --model "$MODEL" --save-result)

echo "===== [$LABEL] Scenario A: decode-heavy (in=128 out=256, 32 prompts) ====="
.venv/bin/vllm bench serve "${COMMON[@]}" \
  --random-input-len 128 --random-output-len 256 --num-prompts 32 \
  --result-dir "$OUT" --result-filename "${LABEL}_decode.json" \
  2>&1 | grep -iE "throughput|TTFT|ITL|Successful|elapsed" | tail -12

echo
echo "===== [$LABEL] Scenario B: long-prefill (in=4096 out=16, 64 prompts) ====="
.venv/bin/vllm bench serve "${COMMON[@]}" \
  --random-input-len 4096 --random-output-len 16 --num-prompts 64 \
  --result-dir "$OUT" --result-filename "${LABEL}_prefill.json" \
  2>&1 | grep -iE "throughput|TTFT|ITL|Successful|elapsed" | tail -12

echo
echo "Results saved to $OUT/${LABEL}_{decode,prefill}.json"
