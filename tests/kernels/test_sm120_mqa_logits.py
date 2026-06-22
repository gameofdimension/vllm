# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the sm_120 c4 MQA indexer scorers.

Op1 = paged decode scorer (`fp8_paged_mqa_logits_sm120_triton`).
Op2 = dense prefill scorer (`fp8_mqa_logits_sm120_triton`).

The Triton kernels are checked against the original chunked-PyTorch reference
implementations below (`ref_*`). These references lived in the production tree
originally; once serving became Triton-only they were moved here to serve as
correctness oracles (they are too slow to serve under load).

Both sides run on identical fp8 inputs, so the only difference is MMA /
accumulation rounding (exp vs exp2, fp8 tl.dot vs fp8->f32 dot). Measured max
abs err is ~1.4e-6 on f32 logits; atol=5e-6 (rtol=0) is a ~3.5x margin.
"""
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fp8_mqa_logits_triton import (
    fp8_mqa_logits_sm120_triton,
    fp8_paged_mqa_logits_sm120_triton,
)

H_IDX, D_IDX, BLOCK = 64, 128, 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCORER_ATOL, SCORER_RTOL = 5e-6, 0.0  # measured abs err ~1.4e-6; rtol=0 -> pure atol


# --------------------------------------------------------------------------- #
# PyTorch reference (oracle) — verbatim from the former production module
# --------------------------------------------------------------------------- #
_PAGE_CHUNK = 64


def ref_fp8_paged_mqa_logits(
    q, kv_cache, weights, context_lens, block_tables,
    schedule_metadata: Any, max_model_len: int, clean_logits: bool,
) -> torch.Tensor:
    """[Bn, max_model_len] f32 c4 sparse indexer logits (paged decode)."""
    _ = schedule_metadata
    q_values, q_scale = q
    assert q_scale is None, "FP8 path only"
    B, next_n, H, D = q_values.shape
    num_blocks, block_size, _, hw = kv_cache.shape
    Bn = B * next_n
    device = q_values.device
    qf = q_values.reshape(Bn, H, D).to(torch.float32)
    w = weights.reshape(Bn, H)
    sl = context_lens.reshape(-1).to(torch.int32)
    pt = block_tables
    if pt.shape[0] != Bn:
        pt = pt.unsqueeze(1).expand(-1, next_n, -1).reshape(Bn, -1)
    pt = pt.contiguous()
    max_pages = pt.shape[1]
    kv = kv_cache.view(num_blocks, block_size, hw)
    logits = q_values.new_empty((Bn, max_model_len), dtype=torch.float32)
    logits.fill_(float("-inf"))
    for start in range(0, max_pages, _PAGE_CHUNK):
        end = min(start + _PAGE_CHUNK, max_pages)
        n_pages = end - start
        tok0 = start * block_size
        if tok0 >= max_model_len:
            break
        pt_c = pt[:, start:end]
        gathered = kv[pt_c.clamp(min=0)]
        kv_val = gathered[..., :D].contiguous().view(torch.float8_e4m3fn).to(torch.float32)
        kv_val = kv_val.reshape(Bn, n_pages * block_size, D)
        kv_sc = gathered[..., D:].contiguous().view(torch.float32).reshape(Bn, n_pages * block_size)
        sc = torch.bmm(kv_val, qf.transpose(1, 2))
        sc = F.relu(sc)
        sc = sc * w.unsqueeze(1)
        sc = sc.sum(dim=2)
        sc = sc * kv_sc
        out_w = min(n_pages * block_size, max_model_len - tok0)
        if out_w > 0:
            logits[:, tok0: tok0 + out_w] = sc[:, :out_w]
    positions = torch.arange(max_model_len, device=device)
    logits.masked_fill_(positions.unsqueeze(0) >= sl.unsqueeze(1), float("-inf"))
    return logits


_K_CHUNK = 2048


def ref_fp8_mqa_logits(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits: bool,
) -> torch.Tensor:
    """[M, N] f32 c4 sparse indexer logits (dense prefill)."""
    q_values, q_scale = q
    assert q_scale is None, "FP8 path only"
    k_packed, k_scales = kv
    M, H, D = q_values.shape
    N = k_packed.shape[0]
    device = q_values.device
    qf = q_values.to(torch.float32)
    w = weights
    kf = k_packed.to(torch.float32)
    ksc = k_scales.contiguous().view(torch.float32).reshape(N)
    logits = q_values.new_empty((M, N), dtype=torch.float32)
    ks = cu_seqlen_ks.reshape(M)
    ke = cu_seqlen_ke.reshape(M)
    n_max = min(int(ke.max().item()), N) if N > 0 else 0
    for s in range(0, n_max, _K_CHUNK):
        e = min(s + _K_CHUNK, n_max)
        kc = kf[s:e]
        ksc_c = ksc[s:e]
        sc = torch.einsum("mhd,cd->mch", qf, kc)
        sc = F.relu(sc)
        sc = sc * w.unsqueeze(1)
        sc = sc.sum(dim=2)
        sc = sc * ksc_c
        logits[:, s:e] = sc
    if n_max < N:
        logits[:, n_max:] = float("-inf")
    n_idx = torch.arange(N, device=device)
    valid = (n_idx.unsqueeze(0) >= ks.unsqueeze(1)) & (n_idx.unsqueeze(0) < ke.unsqueeze(1))
    logits.masked_fill_(~valid, float("-inf"))
    return logits


# --------------------------------------------------------------------------- #
# input generators (per-token KV layout: 128B fp8 value + 4B f32 scale)
# --------------------------------------------------------------------------- #
def _make_paged(B, context_len, next_n=1, H=H_IDX, D=D_IDX, block_size=BLOCK):
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
    kv_cache = kv.view(num_blocks, block_size, 1, D + 4)
    bt = torch.zeros(B, pages_per_row, device=DEV, dtype=torch.int32)
    for r in range(B):
        base = r * pages_per_row
        bt[r] = torch.arange(base, base + pages_per_row, device=DEV, dtype=torch.int32)
    bt = bt.unsqueeze(1).expand(-1, next_n, -1).reshape(B * next_n, -1).contiguous()
    return q_vals, kv_cache, weights, context_lens, bt


def _make_prefill(M, N, H=H_IDX, D=D_IDX):
    q_vals = (torch.randn(M, H, D, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    k_packed = (torch.randn(N, D, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    k_scales = torch.rand(N, device=DEV, dtype=torch.float32) * 2.0 + 0.5
    weights = torch.randn(M, H, device=DEV, dtype=torch.float32)
    edges = torch.linspace(0, N, M + 1, device=DEV).to(torch.int32)
    return q_vals, (k_packed, k_scales), weights, edges[:-1].contiguous(), edges[1:].contiguous()


def _assert_close(tri, ref, name):
    inf = torch.isinf(ref)
    assert torch.equal(inf, torch.isinf(tri)), f"{name}: -inf mask mismatch"
    sel = ~inf
    if sel.any():
        torch.testing.assert_close(tri[sel], ref[sel], rtol=SCORER_RTOL, atol=SCORER_ATOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("B,ctx", [(1, 4096), (2, 32768), (4, 131072)])
def test_paged_decode_scorer(B, ctx):
    q, kv, w, cl, bt = _make_paged(B, ctx)
    args = ((q, None), kv, w, cl, bt, None, ctx, False)
    ref = ref_fp8_paged_mqa_logits(*args)
    tri = fp8_paged_mqa_logits_sm120_triton(*args)
    _assert_close(tri, ref, f"paged B={B} ctx={ctx}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("M,N", [(64, 4096), (256, 32768)])
def test_dense_prefill_scorer(M, N):
    q, kv, w, ks, ke = _make_prefill(M, N)
    args = ((q, None), kv, w, ks, ke, False)
    ref = ref_fp8_mqa_logits(*args)
    tri = fp8_mqa_logits_sm120_triton(*args)
    _assert_close(tri, ref, f"prefill M={M} N={N}")
