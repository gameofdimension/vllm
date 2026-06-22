# SPDX-License-Identifier: Apache-2.0
"""sm_120 MLA decode — two-phase approach (gather+dequant → dense attention).

Replaces the fused Triton kernel (gather-inside-attention-loop, 88.8% of decode
GPU time) with SGLang's newer architecture: separate the gather from the attention.
Phase 1: PyTorch fancy-index gather + fp8 dequant → contiguous [B, topk, D] bf16.
Phase 2: cuBLAS einsum dense attention on the contiguous buffer.

The gather is done ONCE per batch (shared across all H heads), eliminating the
H-fold redundant gather of the fused kernel. The attention runs on contiguous
data via tensor-core einsums.
"""
from typing import Optional, Tuple

import torch

LOG2E = 1.4426950408889634
_NOPE_DIM = 448
_ROPE_DIM = 64
_D = _NOPE_DIM + _ROPE_DIM  # 512


def _gather_dequant_one(k_cache, indices, topk_length, total_tokens):
    """Gather+dequant one KV cache → [T, topk, D] f32 + [T, topk] bool invalid."""
    num_pages, page_size = k_cache.shape[0], k_cache.shape[1]
    topk = indices.shape[-1]
    idx_flat = indices.reshape(total_tokens, topk)
    safe = idx_flat.clamp(min=0)
    pid = safe // page_size
    poff = safe % page_size
    # Flat uint8 view per page: [num_pages, page_bytes]
    raw = k_cache.view(torch.uint8).reshape(num_pages, -1)
    bpp = raw.shape[1]  # bytes per page
    ps576 = page_size * 576
    data = raw[:, :ps576].reshape(num_pages, page_size, 576)
    scales = raw[:, ps576:ps576 + page_size * 8].reshape(num_pages, page_size, 8)
    gd = data[pid, poff]      # [T, topk, 576]
    gs = scales[pid, poff]    # [T, topk, 8]
    # nope fp8 → f32 with ue8m0 scales (7 groups × 64)
    nope = gd[..., :_NOPE_DIM].contiguous().view(torch.float8_e4m3fn).reshape(
        total_tokens, topk, 7, 64)
    sc = gs[..., :7].to(torch.float32).reshape(total_tokens, topk, 7, 1)
    kv_nope = (nope.to(torch.float32) * torch.exp2(sc - 127.0)).reshape(
        total_tokens, topk, _NOPE_DIM)
    # rope bf16
    kv_rope = gd[..., _NOPE_DIM:_NOPE_DIM + _ROPE_DIM * 2].contiguous().view(
        torch.bfloat16).to(torch.float32).reshape(total_tokens, topk, _ROPE_DIM)
    kv = torch.cat([kv_nope, kv_rope], dim=-1)  # [T, topk, D] f32
    # invalid mask
    invalid = idx_flat < 0
    if topk_length is not None:
        pos = torch.arange(topk, device=k_cache.device)
        invalid = invalid | (pos.unsqueeze(0) >= topk_length.reshape(-1, 1))
    return kv, invalid


def flash_mla_sparse_decode_two_phase(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    head_dim_v: int,
    softmax_scale: float,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two-phase sparse MLA decode. Same signature as flash_mla_sparse_decode_triton."""
    B, _, H, D = q.shape
    total_tokens = B  # s_q = 1 for decode
    if softmax_scale is None:
        softmax_scale = D ** (-0.5)

    kv, mask = _gather_dequant_one(k_cache, indices, topk_length, total_tokens)
    if extra_k_cache is not None and extra_indices is not None:
        kv2, mask2 = _gather_dequant_one(
            extra_k_cache, extra_indices, extra_topk_length, total_tokens)
        kv = torch.cat([kv, kv2], dim=1)
        mask = torch.cat([mask, mask2], dim=1)

    topk_total = kv.shape[1]
    qf = q.squeeze(1).to(torch.float32) * softmax_scale  # [B, H, D]
    scores = torch.einsum("bhd,bvd->bhv", qf, kv)        # [B, H, topk_total]
    scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
    scores_max = scores.amax(-1, keepdim=True)            # [B, H, 1]
    e = torch.exp(scores - scores_max)
    numer = torch.einsum("bhv,bvd->bhd", e, kv)          # [B, H, D]
    denom = e.sum(-1, keepdim=True)                        # [B, H, 1]
    if attn_sink is not None:
        sink = attn_sink
        sink_b = sink.reshape(1, 1, 1) if sink.dim() == 0 else sink.reshape(1, -1, 1)
        denom = denom + torch.exp(sink_b.to(torch.float32) - scores_max)
    out = numer / denom.clamp(min=1e-20)                  # [B, H, D]
    lse = (scores_max + torch.log(denom.clamp(min=1e-20))).squeeze(-1)  # [B, H]
    return out.to(torch.bfloat16).unsqueeze(1), lse.unsqueeze(1)
