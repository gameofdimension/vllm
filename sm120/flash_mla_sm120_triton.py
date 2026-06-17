# SPDX-License-Identifier: Apache-2.0
# Ported verbatim from SGLang (sgl-project/sglang, main):
#   python/sglang/srt/layers/attention/flash_mla_sm120_triton.py
# SM120-optimized Triton FlashMLA sparse decode kernel for DeepSeek-V4 MLA.
# Self-contained (pure Triton + torch), no DeepGEMM. See sm120/CHANGES.md.
"""SM120-optimized Triton FlashMLA sparse decode kernel — Tiled V2.

Replaces V1's serial token loop with a tiled vectorized approach:
  1. BLOCK_T tokens loaded simultaneously via 2D gather (vs 1-at-a-time)
  2. All BLOCK_T QK scores computed at once via vectorized mul-reduce
  3. V accumulation via vectorized weighted sum across BLOCK_T tokens
  4. Online softmax operates on tile-level maxima (fewer rescales)

Three typed views of the same paged buffer handle FP8/uint8/BF16 regions:
- float8_e4m3fn view -> nope FP8 values (direct load + dequant)
- uint8 view -> UE8M0 scale bytes (raw integer -> exp2 conversion)
- bfloat16 view -> rope BF16 values (direct load)

DSv4 page layout (per token, 576 bytes data + 8 bytes scales):
  Data section: [0:448] FP8 nope | [448:576] BF16 rope (64 values = 128 bytes)
  Scale section: [page_size*576 + offset*8 : +7] UE8M0 scales (7 groups of 64)

Target: RTX PRO 6000/5000 (SM120, ~188/150 SMs, 99KB SMEM, no TMEM/tcgen05).
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)

# DSv4 KV cache layout constants
_NOPE_DIM = 448
_ROPE_DIM = 64
_D = _NOPE_DIM + _ROPE_DIM  # 512
_TOKEN_DATA_STRIDE = 576  # bytes per token in data section
_SCALE_STRIDE = 8  # bytes per token in scale section


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_T": 32}, num_warps=8, num_stages=2),
    ],
    key=["topk_rounded"],
)
@triton.jit
def _tiled_sparse_decode_kernel(
    Q_ptr,
    cache_fp8_ptr,
    cache_uint8_ptr,
    cache_bf16_ptr,
    indices_ptr,
    topk_len_ptr,
    O_ptr,
    LSE_ptr,
    sm_scale: tl.float32,
    page_size: tl.int32,
    page_bytes: tl.int64,
    scale_section_off: tl.int64,
    H: tl.int32,
    topk: tl.int32,
    topk_rounded: tl.int32,
    has_topk_len: tl.constexpr,
    stride_qb: tl.int32,
    stride_qh: tl.int32,
    stride_ob: tl.int32,
    stride_oh: tl.int32,
    stride_ib: tl.int32,
    NOPE_PAD: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_DIM_RT: tl.int32,
    BLOCK_T: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    q_base = bid * stride_qb + hid * stride_qh
    nope_offs = tl.arange(0, NOPE_PAD)
    nope_mask = nope_offs < NOPE_DIM_RT
    rope_offs = tl.arange(0, ROPE_DIM)

    q_nope = tl.load(Q_ptr + q_base + nope_offs, mask=nope_mask, other=0.0)
    q_nope = q_nope.to(tl.float32) * sm_scale
    q_rope = tl.load(Q_ptr + q_base + NOPE_DIM_RT + rope_offs)
    q_rope = q_rope.to(tl.float32) * sm_scale

    valid_topk = topk
    if has_topk_len:
        valid_topk = tl.load(topk_len_ptr + bid).to(tl.int32)
        valid_topk = tl.minimum(valid_topk, topk)

    m_i: tl.float32 = -1e30
    l_i: tl.float32 = 0.0
    acc_nope = tl.zeros([NOPE_PAD], dtype=tl.float32)
    acc_rope = tl.zeros([ROPE_DIM], dtype=tl.float32)

    group_ids = (nope_offs // 64).to(tl.int64)
    t_offs = tl.arange(0, BLOCK_T)

    for tile_start in range(0, topk, BLOCK_T):
        t_idx = tile_start + t_offs
        t_in_bounds = t_idx < topk
        t_valid = t_idx < valid_topk

        raw_indices = tl.load(
            indices_ptr + bid * stride_ib + t_idx,
            mask=t_in_bounds,
            other=-1,
        )
        idx_valid = t_valid & (raw_indices >= 0)

        safe_indices = tl.where(idx_valid, raw_indices, tl.zeros_like(raw_indices))
        page_ids = (safe_indices // page_size).to(tl.int64)
        page_offs_t = (safe_indices % page_size).to(tl.int64)
        token_data_bases = page_ids * page_bytes + page_offs_t * 576

        nope_addrs = token_data_bases[:, None] + nope_offs[None, :].to(tl.int64)
        nope_2d_mask = idx_valid[:, None] & nope_mask[None, :]
        kv_nope_fp8 = tl.load(
            cache_fp8_ptr + nope_addrs,
            mask=nope_2d_mask,
            other=0.0,
        )

        scale_bases = page_ids * page_bytes + scale_section_off + page_offs_t * 8
        scale_addrs = scale_bases[:, None] + group_ids[None, :]
        scale_raw = tl.load(
            cache_uint8_ptr + scale_addrs,
            mask=nope_2d_mask,
            other=127,
        )
        scale_f32 = tl.math.exp2(scale_raw.to(tl.float32) - 127.0)
        kv_nope = tl.where(nope_2d_mask, kv_nope_fp8.to(tl.float32) * scale_f32, 0.0)

        rope_byte_bases = token_data_bases + 448
        rope_elem_bases = (rope_byte_bases // 2).to(tl.int64)
        rope_addrs = rope_elem_bases[:, None] + rope_offs[None, :].to(tl.int64)
        kv_rope = tl.load(
            cache_bf16_ptr + rope_addrs,
            mask=idx_valid[:, None],
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(q_nope[None, :] * kv_nope, axis=1) + tl.sum(
            q_rope[None, :] * kv_rope, axis=1
        )
        scores = tl.where(idx_valid, scores, -1e30)

        scores_log2 = scores * LOG2E
        tile_max = tl.max(scores_log2)
        m_new = tl.maximum(m_i, tile_max)

        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(scores_log2 - m_new)
        p = tl.where(idx_valid, p, 0.0)

        l_i = l_i * alpha + tl.sum(p)

        acc_nope = acc_nope * alpha + tl.sum(p[:, None] * kv_nope, axis=0)
        acc_rope = acc_rope * alpha + tl.sum(p[:, None] * kv_rope, axis=0)
        m_i = m_new

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    acc_nope = acc_nope / safe_l
    acc_rope = acc_rope / safe_l

    lse = tl.where(l_i > 0.0, m_i / LOG2E + tl.math.log(safe_l), float("-inf"))

    o_base = bid * stride_ob + hid * stride_oh
    tl.store(O_ptr + o_base + nope_offs, acc_nope.to(tl.bfloat16), mask=nope_mask)
    tl.store(O_ptr + o_base + NOPE_DIM_RT + rope_offs, acc_rope.to(tl.bfloat16))
    tl.store(LSE_ptr + bid * H + hid, lse)


def _run_triton_sparse_decode(
    q: torch.Tensor,  # [B, 1, H, D] bf16
    k_cache: torch.Tensor,  # [num_pages, page_size, 1, bpt] float8
    indices: torch.Tensor,  # [B, ...] int32
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, _, H, D = q.shape
    num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]
    page_bytes = k_cache.stride(0)

    flat_indices = indices.reshape(B, -1).contiguous()
    topk = flat_indices.shape[1]

    total_elems = num_pages * page_bytes
    raw_flat = k_cache.as_strided((total_elems,), (1,))
    raw_uint8 = raw_flat.view(torch.uint8)
    raw_fp8 = raw_uint8.view(torch.float8_e4m3fn)
    raw_bf16 = raw_uint8.view(torch.bfloat16)

    q3 = q.squeeze(1)
    if not q3.is_contiguous():
        q3 = q3.contiguous()

    out = torch.zeros(B, H, D, dtype=torch.bfloat16, device=q.device)
    lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device=q.device)

    topk_rounded = triton.next_power_of_2(topk)

    grid = (B, H)
    _tiled_sparse_decode_kernel[grid](
        q3,
        raw_fp8,
        raw_uint8,
        raw_bf16,
        flat_indices,
        (
            topk_length
            if topk_length is not None
            else torch.empty(0, device=q.device, dtype=torch.int32)
        ),
        out,
        lse,
        softmax_scale,
        page_size,
        int(page_bytes),
        int(page_size * _TOKEN_DATA_STRIDE),
        H,
        topk,
        topk_rounded,
        topk_length is not None,
        q3.stride(0),
        q3.stride(1),
        out.stride(0),
        out.stride(1),
        flat_indices.stride(0),
        NOPE_PAD=512,
        ROPE_DIM=_ROPE_DIM,
        NOPE_DIM_RT=_NOPE_DIM,
    )

    return out.unsqueeze(1), lse.unsqueeze(1)


def _merge_partial_attn(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_lse = torch.maximum(lse1, lse2)
    w1 = torch.where(lse1 > -1e20, torch.exp(lse1 - max_lse), torch.zeros_like(lse1))
    w2 = torch.where(lse2 > -1e20, torch.exp(lse2 - max_lse), torch.zeros_like(lse2))
    total = (w1 + w2).clamp(min=1e-20)
    merged = (
        w1.unsqueeze(-1) * out1.float() + w2.unsqueeze(-1) * out2.float()
    ) / total.unsqueeze(-1)
    merged_lse = max_lse + torch.log(total)
    return merged.to(torch.bfloat16), merged_lse


def _apply_attn_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sink_lse = attn_sink.view(1, 1, -1).expand_as(lse)
    combined_lse = torch.logaddexp(lse, sink_lse)
    w = torch.where(
        lse > -1e20,
        torch.exp(lse - combined_lse),
        torch.zeros_like(lse),
    )
    return (out.float() * w.unsqueeze(-1)).to(torch.bfloat16), combined_lse


def flash_mla_sparse_decode_triton(
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
    """SM120-optimized sparse MLA decode using tiled Triton kernel.

    Processes SWA and extra (c4/c128) caches separately via the same
    Triton kernel, then merges results using LSE-weighted combination.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    out, lse = _run_triton_sparse_decode(
        q,
        k_cache,
        indices,
        topk_length,
        softmax_scale,
    )

    if extra_k_cache is not None and extra_indices is not None:
        out_extra, lse_extra = _run_triton_sparse_decode(
            q,
            extra_k_cache,
            extra_indices,
            extra_topk_length,
            softmax_scale,
        )
        out, lse = _merge_partial_attn(out, lse, out_extra, lse_extra)

    if attn_sink is not None:
        out, lse = _apply_attn_sink(out, lse, attn_sink)

    return out, lse.permute(0, 2, 1)
