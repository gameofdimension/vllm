#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Standalone micro-bench for the sm_120 c4 indexer scorers + MLA prefill.

Imports the LIVE editable vllm tree. Runs WITHOUT the server: constructs
realistic inputs, times the current PyTorch oracle for each op, and (once a
Triton kernel exists) compares correctness + speedup.

Usage:
    .venv/bin/python sm120/bench/scorer_bench.py                 # baseline only
    .venv/bin/python sm120/bench/scorer_bench.py --op paged      # just Op1
    .venv/bin/python sm120/bench/scorer_bench.py --compare       # also run Triton + diff
"""
from __future__ import annotations

import argparse
import statistics
import sys

import torch

# Live editable tree.
from vllm.model_executor.layers.fp8_paged_mqa_logits_sm120 import (
    fp8_mqa_logits_sm120,
    fp8_paged_mqa_logits_sm120,
)
from vllm.models.deepseek_v4.nvidia.flash_mla_sparse_fwd_sm120 import (
    flash_mla_sparse_fwd_sm120,
)
from vllm.third_party.triton_kernels.testing import assert_close

DEV = torch.device("cuda:0")
H_IDX = 64           # indexer heads
D_IDX = 128          # indexer head dim
BLOCK_SIZE = 64      # page block size
D_MLA = 512          # MLA head dim


# --------------------------------------------------------------------------- #
# input generators
# --------------------------------------------------------------------------- #
def make_paged_decode_inputs(B, context_len, next_n=1, H=H_IDX, D=D_IDX,
                             block_size=BLOCK_SIZE):
    """Op1 inputs. kv_cache per-token layout: 128B fp8 value + 4B f32 scale."""
    num_blocks = B * ((context_len + block_size - 1) // block_size)
    pages_per_row = (context_len + block_size - 1) // block_size

    # query [B, next_n, H, D] fp8
    q_vals = (torch.randn(B, next_n, H, D, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    weights = torch.randn(B * next_n, H, device=DEV, dtype=torch.float32)
    context_lens = torch.full((B, next_n), context_len, device=DEV, dtype=torch.int32)

    # kv_cache [num_blocks, block_size, 1, D+4] uint8
    vals = (torch.randn(num_blocks, block_size, D, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    scales = (torch.rand(num_blocks, block_size, device=DEV, dtype=torch.float32) * 2.0 + 0.5)
    kv = torch.empty(num_blocks, block_size, D + 4, device=DEV, dtype=torch.uint8)
    kv[:, :, :D] = vals.view(torch.uint8)
    kv[:, :, D:] = scales.view(torch.uint8).reshape(num_blocks, block_size, 4)
    kv_cache = kv.view(num_blocks, block_size, 1, D + 4)

    # block_tables [B, max_blocks] -> distinct real blocks per row
    bt = torch.zeros(B, pages_per_row, device=DEV, dtype=torch.int32)
    for r in range(B):
        base = r * pages_per_row
        bt[r] = torch.arange(base, base + pages_per_row, device=DEV, dtype=torch.int32)
    bt = bt.unsqueeze(1).expand(-1, next_n, -1).reshape(B * next_n, -1).contiguous()
    return q_vals, kv_cache, weights, context_lens, bt


def make_prefill_inputs(M, N, H=H_IDX, D=D_IDX):
    """Op2 inputs: dense gathered KV + per-query [ks, ke) ranges tiling [0, N)."""
    q_vals = (torch.randn(M, H, D, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    k_packed = (torch.randn(N, D, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    k_scales = torch.rand(N, device=DEV, dtype=torch.float32) * 2.0 + 0.5
    weights = torch.randn(M, H, device=DEV, dtype=torch.float32)
    # split [0, N) into M contiguous chunks
    edges = torch.linspace(0, N, M + 1, device=DEV).to(torch.int32)
    cu_ks = edges[:-1].contiguous()
    cu_ke = edges[1:].contiguous()
    return q_vals, (k_packed, k_scales), weights, cu_ks, cu_ke


def make_mla_prefill_inputs(Tq, topk, H=H_IDX, D=D_MLA):
    """Op3 inputs: gathered sparse MLA attention over dense bf16 KV."""
    pool = topk * 2
    q = (torch.randn(Tq, H, D, device=DEV, dtype=torch.bfloat16) * 0.05)
    kv = (torch.randn(pool, 1, D, device=DEV, dtype=torch.bfloat16) * 0.05)
    indices = torch.randint(0, pool, (Tq, 1, topk), device=DEV, dtype=torch.int32)
    topk_length = torch.full((Tq,), topk, device=DEV, dtype=torch.int32)
    out = torch.empty(Tq, H, D, device=DEV, dtype=torch.bfloat16)
    sm_scale = 1.0 / (D ** 0.5)
    sink = torch.tensor(-1.0, device=DEV)  # scalar-tensor sink (oracle expects tensor|None)
    return q, kv, indices, sm_scale, sink, topk_length, out


# --------------------------------------------------------------------------- #
# timing + compare
# --------------------------------------------------------------------------- #
def time_fn(fn, args, warmup=5, iters=20):
    torch.cuda.synchronize(DEV)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize(DEV)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(*args)
        stops[i].record()
    torch.cuda.synchronize(DEV)
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, stops))


def fmt_ms(ms):
    return f"{ms:8.3f} ms" if ms is not None else "    OOM  "


# --------------------------------------------------------------------------- #
# op drivers
# --------------------------------------------------------------------------- #
def run_paged(shapes, compare):
    print("\n=== Op1: paged decode c4 scorer (oracle = fp8_paged_mqa_logits_sm120) ===")
    print(f"{'B':>4} {'ctx':>9} {'oracle':>10} {'triton':>10} {'speedup':>8}  {'max_err':>9}  status")
    for B, ctx in shapes:
        try:
            q, kv, w, cl, bt = make_paged_decode_inputs(B, ctx)
            mml = ctx  # measure scorer compute over the actual context
            a = ((q, None), kv, w, cl, bt, None, mml, False)
            ref = fp8_paged_mqa_logits_sm120(*a)
            t_ref = time_fn(fp8_paged_mqa_logits_sm120, a)
        except torch.OutOfMemoryError:
            print(f"{B:>4} {ctx:>9} {'-':>10}")
            continue
        line = f"{B:>4} {ctx:>9} {fmt_ms(t_ref):>10}"
        if compare:
            try:
                from vllm.model_executor.layers.fp8_mqa_logits_triton import (
                    fp8_paged_mqa_logits_sm120_triton,
                )
                tri = fp8_paged_mqa_logits_sm120_triton(*a)
                assert_close(ref, tri, description=f"Op1 B={B} ctx={ctx}")
                t_tri = time_fn(fp8_paged_mqa_logits_sm120_triton, a)
                err = (tri - ref).abs()
                err = err[~torch.isinf(err) & ~torch.isnan(err)].max().item()
                line += f" {fmt_ms(t_tri):>10} {t_ref / t_tri:>7.2f}x  {err:>9.2e}  OK"
            except Exception as e:  # noqa: BLE001
                line += f" {'-':>10} {'-':>8}  {'-':>9}  FAIL: {type(e).__name__}: {e}"
        print(line)


def run_prefill(shapes, compare):
    print("\n=== Op2: non-paged prefill c4 scorer (oracle = fp8_mqa_logits_sm120) ===")
    print(f"{'M':>5} {'N':>7} {'oracle':>10} {'triton':>10} {'speedup':>8}  {'max_err':>9}  status")
    for M, N in shapes:
        try:
            q, kv, w, ks, ke = make_prefill_inputs(M, N)
            a = ((q, None), kv, w, ks, ke, False)
            ref = fp8_mqa_logits_sm120(*a)
            t_ref = time_fn(fp8_mqa_logits_sm120, a)
        except torch.OutOfMemoryError:
            print(f"{M:>5} {N:>7} {'OOM':>10}")
            continue
        line = f"{M:>5} {N:>7} {fmt_ms(t_ref):>10}"
        if compare:
            try:
                from vllm.model_executor.layers.fp8_mqa_logits_triton import (
                    fp8_mqa_logits_sm120_triton,
                )
                tri = fp8_mqa_logits_sm120_triton(*a)
                assert_close(ref, tri, description=f"Op2 M={M} N={N}")
                t_tri = time_fn(fp8_mqa_logits_sm120_triton, a)
                err = (tri - ref).abs()
                err = err[~torch.isinf(err) & ~torch.isnan(err)].max().item()
                line += f" {fmt_ms(t_tri):>10} {t_ref / t_tri:>7.2f}x  {err:>9.2e}  OK"
            except Exception as e:  # noqa: BLE001
                line += f" {'-':>10} {'-':>8}  {'-':>9}  FAIL: {type(e).__name__}: {e}"
        print(line)


def run_mla_prefill(shapes, compare):
    print("\n=== Op3: MLA prefill gathered sparse (oracle = flash_mla_sparse_fwd_sm120) ===")
    print(f"{'Tq':>5} {'topk':>5} {'oracle':>10} {'triton':>10} {'speedup':>8}  {'max_err':>9}  status")
    for Tq, topk in shapes:
        try:
            q, kv, idx, sms, sink, topk_length, out = make_mla_prefill_inputs(Tq, topk)
            a = (q, kv, idx, sms, sink, topk_length, out)
            ref = flash_mla_sparse_fwd_sm120(*a)
            t_ref = time_fn(flash_mla_sparse_fwd_sm120, a)
        except torch.OutOfMemoryError:
            print(f"{Tq:>5} {topk:>5} {'OOM':>10}")
            continue
        line = f"{Tq:>5} {topk:>5} {fmt_ms(t_ref):>10}"
        if compare:
            try:
                from vllm.models.deepseek_v4.nvidia.flash_mla_sparse_prefill_triton import (
                    flash_mla_sparse_fwd_sm120_triton,
                )
                ref_saved = ref.clone()
                tri = flash_mla_sparse_fwd_sm120_triton(*a)
                assert_close(ref_saved.to(torch.float32), tri.to(torch.float32),
                             description=f"Op3 Tq={Tq} topk={topk}")
                t_tri = time_fn(flash_mla_sparse_fwd_sm120_triton, a)
                err = (tri.to(torch.float32) - ref_saved.to(torch.float32)).abs().max().item()
                line += f" {fmt_ms(t_tri):>10} {t_ref / t_tri:>7.2f}x  {err:>9.2e}  OK"
            except Exception as e:  # noqa: BLE001
                line += f" {'-':>10} {'-':>8}  {'-':>9}  FAIL: {type(e).__name__}: {e}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["paged", "prefill", "mla", "all"], default="all")
    ap.add_argument("--compare", action="store_true",
                    help="also run the Triton Op1 kernel and diff vs oracle")
    args = ap.parse_args()

    print(f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"devcap={torch.cuda.get_device_capability(DEV)}")
    torch.cuda.set_device(DEV)

    paged_shapes = [(1, 4096), (1, 32768), (4, 32768), (8, 131072),
                    (32, 32768), (8, 1048576)]
    prefill_shapes = [(512, 4096), (2048, 32768), (4096, 32768), (8192, 32768)]
    mla_shapes = [(512, 512), (2048, 2048), (4096, 2048)]

    if args.op in ("paged", "all"):
        run_paged(paged_shapes, args.compare)
    if args.op in ("prefill", "all"):
        run_prefill(prefill_shapes, args.compare)
    if args.op in ("mla", "all"):
        run_mla_prefill(mla_shapes, args.compare)


if __name__ == "__main__":
    sys.exit(main())
