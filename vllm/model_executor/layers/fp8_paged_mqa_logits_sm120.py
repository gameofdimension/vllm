# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang's fp8_paged_mqa_logits_torch_sm120
# (sgl-project/sglang: python/sglang/srt/layers/attention/dsv4/indexer.py).
#
# Differences from the SGLang original:
# 1. vLLM's indexer KV cache is laid out PER-TOKEN as
#    [num_blocks, block_size, 1, head_dim+4] (128B fp8 value + 4B f32 scale per
#    token), whereas SGLang's torch scorer assumes a BLOCK-LEVEL split (all
#    values then all scales). This version gathers/splits per-token.
# 2. CHUNKED gather/scatter: the full materialization [Bn, max_pages, block_size,
#    D] fp32 OOMs for long contexts (max_model_len=1M => 16384 pages). We stream
#    PAGE_CHUNK pages at a time into the [Bn, max_model_len] output, bounding
#    intermediate memory. This op runs in the breakable-cudagraph eager segment,
#    so a Python loop is acceptable.
#
# FP8 indexer path only (q_scale folded into `weights`); FP4 indexer not
# supported on sm_120 (matches SGLang). See sm120/CHANGES.md.
"""sm_120 FP8 paged-MQA indexer logits (chunked PyTorch, eager-segment safe).

Replaces vLLM's DeepGEMM `fp8_fp4_paged_mqa_logits` for the c4 indexer on sm_120.
Signature matches fp8_fp4_paged_mqa_logits so it can be swapped in directly.
"""
from typing import Any

import torch
import torch.nn.functional as F

# Pages processed per chunk. Bounds intermediate memory to
# [Bn, PAGE_CHUNK*block_size, D] fp32 (e.g. 128 * 64*64 * 128 * 4 B ~ 256 MiB).
_PAGE_CHUNK = 64


def fp8_paged_mqa_logits_sm120(
    q,  # tuple (q_values [B, next_n, H, D] fp8, q_scale None for FP8)
    kv_cache,  # [num_blocks, block_size, 1, D+4] uint8 (per-token 128B val + 4B f32 scale)
    weights,  # [B*next_n, H] f32 (q scale folded in on FP8 path)
    context_lens,  # [B, next_n] or [B,1] int32
    block_tables,  # [B, max_blocks] int32
    schedule_metadata: Any,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """Return logits [B*next_n, max_model_len] f32 for the c4 sparse indexer."""
    _ = schedule_metadata  # DeepGEMM-only; unused
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError(
            "sm_120 indexer scorer: FP4 indexer q (q_scale is not None) is not "
            "supported; the FP8 indexer path is required on sm_120."
        )

    B, next_n, H, D = q_values.shape
    assert D == 128, "DSV4 indexer head_dim must be 128"
    num_blocks, block_size, _, hw = kv_cache.shape
    assert hw == D + 4, f"expected per-token width {D + 4}, got {hw}"

    Bn = B * next_n
    device = q_values.device

    qf = q_values.reshape(Bn, H, D).to(torch.float32)            # [Bn, H, D]
    w = weights.reshape(Bn, H)                                   # [Bn, H]
    sl = context_lens.reshape(-1).to(torch.int32)                # [Bn]

    pt = block_tables
    if pt.shape[0] != Bn:
        pt = pt.unsqueeze(1).expand(-1, next_n, -1).reshape(Bn, -1)
    pt = pt.contiguous()
    max_pages = pt.shape[1]

    kv = kv_cache.view(num_blocks, block_size, hw)               # [num_blocks, block_size, hw]

    logits = q_values.new_empty((Bn, max_model_len), dtype=torch.float32)
    logits.fill_(float("-inf"))

    # Chunked paged gather + score, streamed into the output.
    for start in range(0, max_pages, _PAGE_CHUNK):
        end = min(start + _PAGE_CHUNK, max_pages)
        n_pages = end - start
        tok0 = start * block_size
        if tok0 >= max_model_len:
            break
        pt_c = pt[:, start:end]                                  # [Bn, n_pages]
        gathered = kv[pt_c.clamp(min=0)]                         # [Bn, n_pages, block_size, hw]
        kv_val = (
            gathered[..., :D].contiguous().view(torch.float8_e4m3fn).to(torch.float32)
        )                                                        # [Bn, n_pages, block_size, D]
        kv_val = kv_val.reshape(Bn, n_pages * block_size, D)
        kv_sc = (
            gathered[..., D:].contiguous().view(torch.float32).reshape(Bn, n_pages * block_size)
        )                                                        # [Bn, n_pages*block_size]

        sc = torch.bmm(kv_val, qf.transpose(1, 2))              # [Bn, n_pages*block_size, H]
        sc = F.relu(sc)
        sc = sc * w.unsqueeze(1)
        sc = sc.sum(dim=2)                                       # [Bn, n_tokens_chunk]
        sc = sc * kv_sc

        out_w = min(n_pages * block_size, max_model_len - tok0)
        if out_w > 0:
            logits[:, tok0 : tok0 + out_w] = sc[:, :out_w]

    # Mask positions beyond each batch's context length to -inf.
    positions = torch.arange(max_model_len, device=device)
    invalid = positions.unsqueeze(0) >= sl.unsqueeze(1)
    logits.masked_fill_(invalid, float("-inf"))
    return logits


# Chunk size over the K dimension for the (non-paged) prefill scorer.
_K_CHUNK = 2048


def fp8_mqa_logits_sm120(
    q,             # tuple (q_values [M, H, D] fp8, q_scale None for FP8)
    kv,            # tuple (k_packed [N, D] fp8, k_scales [N] f32)
    weights,       # [M, H] f32 (q scale folded in on FP8 path)
    cu_seqlen_ks,  # [M] int32 (inclusive K-start per query)
    cu_seqlen_ke,  # [M] int32 (exclusive K-end per query)
    clean_logits: bool,
) -> torch.Tensor:
    """sm_120 non-paged FP8 MQA logits for the prefill indexer.

    Replaces DeepGEMM `fp8_fp4_mqa_logits`. For each of M query positions,
    score K positions in [cu_seqlen_ks[m], cu_seqlen_ke[m]) over the gathered
    dense KV. Returns logits [M, N] f32. FP8 path only.
    """
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError(
            "sm_120 prefill indexer scorer: FP4 q not supported."
        )
    k_packed, k_scales = kv

    M, H, D = q_values.shape
    assert D == 128, "DSV4 indexer head_dim must be 128"
    N = k_packed.shape[0]
    device = q_values.device

    qf = q_values.to(torch.float32)                   # [M, H, D]
    w = weights                                       # [M, H]
    kf = k_packed.to(torch.float32)                   # [N, D]
    ksc = k_scales.contiguous().view(torch.float32).reshape(N)  # [N]

    logits = q_values.new_empty((M, N), dtype=torch.float32)

    ks = cu_seqlen_ks.reshape(M)
    ke = cu_seqlen_ke.reshape(M)
    n_max = int(ke.max().item()) if N > 0 else 0
    n_max = min(n_max, N)

    # Chunked score over the K dimension, streamed into [M, N].
    for s in range(0, n_max, _K_CHUNK):
        e = min(s + _K_CHUNK, n_max)
        kc = kf[s:e]                                  # [C, D]
        ksc_c = ksc[s:e]                              # [C]
        sc = torch.einsum("mhd,cd->mch", qf, kc)      # [M, C, H]
        sc = F.relu(sc)
        sc = sc * w.unsqueeze(1)
        sc = sc.sum(dim=2)                            # [M, C]
        sc = sc * ksc_c
        logits[:, s:e] = sc

    if n_max < N:
        logits[:, n_max:] = float("-inf")

    # Mask K positions outside each query's [ks, ke) range.
    n_idx = torch.arange(N, device=device)
    valid = (n_idx.unsqueeze(0) >= ks.unsqueeze(1)) & (n_idx.unsqueeze(0) < ke.unsqueeze(1))
    logits.masked_fill_(~valid, float("-inf"))
    return logits
