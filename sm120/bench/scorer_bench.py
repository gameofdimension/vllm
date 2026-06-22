#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""sm_120 op-level LATENCY bench for the three Triton kernels.

Correctness (Triton vs the PyTorch oracle) now lives in
tests/kernels/test_sm120_{mqa_logits,mla_prefill}.py. This script only measures
Triton kernel latency at representative shapes (quick op-level perf check).
The serving-level benchmark is sm120/bench/run_perf.sh (vLLM-Triton vs SGLang).

Usage:
    .venv/bin/python sm120/bench/scorer_bench.py            # all three ops
    .venv/bin/python sm120/bench/scorer_bench.py --op paged # just Op1
"""
from __future__ import annotations

import argparse
import statistics

import torch

from vllm.model_executor.layers.fp8_mqa_logits_triton import (
    fp8_mqa_logits_sm120_triton,
    fp8_paged_mqa_logits_sm120_triton,
)
from vllm.models.deepseek_v4.nvidia.flash_mla_sparse_prefill_triton import (
    flash_mla_sparse_fwd_sm120_triton,
)

DEV = torch.device("cuda:0")
H_IDX, D_IDX, BLOCK = 64, 128, 64
D_MLA = 512


def make_paged_decode_inputs(B, context_len, next_n=1, H=H_IDX, D=D_IDX,
                             block_size=BLOCK):
    num_blocks = B * ((context_len + block_size - 1) // block_size)
    pages_per_row = (context_len + block_size - 1) // block_size
    q_vals = (torch.randn(B, next_n, H, D, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    weights = torch.randn(B * next_n, H, device=DEV, dtype=torch.float32)
    context_lens = torch.full((B, next_n), context_len, device=DEV, dtype=torch.int32)
    vals = (torch.randn(num_blocks, block_size, D, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    scales = torch.rand(num_blocks, block_size, device=DEV, dtype=torch.float32) * 2.0 + 0.5
    kv = torch.empty(num_blocks, block_size, D + 4, device=DEV, dtype=torch.uint8)
    kv[:, :, :D] = vals.view(torch.uint8)
    kv[:, :, D:] = scales.view(torch.uint8).reshape(num_blocks, block_size, 4)
    bt = torch.zeros(B, pages_per_row, device=DEV, dtype=torch.int32)
    for r in range(B):
        base = r * pages_per_row
        bt[r] = torch.arange(base, base + pages_per_row, device=DEV, dtype=torch.int32)
    bt = bt.unsqueeze(1).expand(-1, next_n, -1).reshape(B * next_n, -1).contiguous()
    return q_vals, kv.view(num_blocks, block_size, 1, D + 4), weights, context_lens, bt


def make_prefill_inputs(M, N, H=H_IDX, D=D_IDX):
    q = (torch.randn(M, H, D, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    k = (torch.randn(N, D, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    ks = torch.rand(N, device=DEV, dtype=torch.float32) * 2.0 + 0.5
    w = torch.randn(M, H, device=DEV, dtype=torch.float32)
    edges = torch.linspace(0, N, M + 1, device=DEV).to(torch.int32)
    return q, (k, ks), w, edges[:-1].contiguous(), edges[1:].contiguous()


def make_mla_prefill_inputs(Tq, topk, H=H_IDX, D=D_MLA):
    pool = topk * 2
    q = torch.randn(Tq, H, D, device=DEV, dtype=torch.bfloat16) * 0.05
    kv = torch.randn(pool, 1, D, device=DEV, dtype=torch.bfloat16) * 0.05
    idx = torch.randint(0, pool, (Tq, 1, topk), device=DEV, dtype=torch.int32)
    tl = torch.full((Tq,), topk, device=DEV, dtype=torch.int32)
    return q, kv, idx, 1.0 / (D ** 0.5), torch.tensor(-1.0, device=DEV), tl


def time_fn(fn, args, warmup=5, iters=20):
    torch.cuda.synchronize(DEV)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize(DEV)
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(iters)]
    for s, e in ev:
        s.record(); fn(*args); e.record()
    torch.cuda.synchronize(DEV)
    return statistics.median(s.elapsed_time(e) for s, e in ev)


def run_paged(shapes):
    print("\nOp1 paged decode scorer (Triton):")
    for B, ctx in shapes:
        try:
            q, kv, w, cl, bt = make_paged_decode_inputs(B, ctx)
            t = time_fn(fp8_paged_mqa_logits_sm120_triton,
                        ((q, None), kv, w, cl, bt, None, ctx, False))
            print(f"  B={B:<3} ctx={ctx:<8} {t:8.3f} ms")
        except torch.OutOfMemoryError:
            print(f"  B={B:<3} ctx={ctx:<8} OOM")


def run_prefill(shapes):
    print("\nOp2 dense prefill scorer (Triton):")
    for M, N in shapes:
        try:
            q, kv, w, ks, ke = make_prefill_inputs(M, N)
            t = time_fn(fp8_mqa_logits_sm120_triton, ((q, None), kv, w, ks, ke, False))
            print(f"  M={M:<5} N={N:<7} {t:8.3f} ms")
        except torch.OutOfMemoryError:
            print(f"  M={M:<5} N={N:<7} OOM")


def run_mla(shapes):
    print("\nOp3 MLA prefill (Triton):")
    for Tq, topk in shapes:
        try:
            q, kv, idx, sms, sink, tl = make_mla_prefill_inputs(Tq, topk)
            out = torch.empty(Tq, H_IDX, D_MLA, device=DEV, dtype=torch.bfloat16)
            t = time_fn(flash_mla_sparse_fwd_sm120_triton, (q, kv, idx, sms, sink, tl, out))
            print(f"  Tq={Tq:<5} topk={topk:<5} {t:8.3f} ms")
        except torch.OutOfMemoryError:
            print(f"  Tq={Tq:<5} topk={topk:<5} OOM")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["paged", "prefill", "mla", "all"], default="all")
    args = ap.parse_args()
    torch.cuda.set_device(DEV)
    print(f"torch={torch.__version__} devcap={torch.cuda.get_device_capability(DEV)}")
    if args.op in ("paged", "all"):
        run_paged([(1, 4096), (1, 32768), (4, 32768), (8, 131072), (8, 1048576)])
    if args.op in ("prefill", "all"):
        run_prefill([(512, 4096), (2048, 32768), (4096, 32768)])
    if args.op in ("mla", "all"):
        run_mla([(512, 512), (2048, 2048)])


if __name__ == "__main__":
    main()
