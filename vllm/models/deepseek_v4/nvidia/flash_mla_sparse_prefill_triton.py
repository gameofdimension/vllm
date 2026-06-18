# SPDX-License-Identifier: Apache-2.0
"""sm_120 fused Triton kernel for DeepSeek-V4 MLA PREFILL gathered sparse attn.

Replaces the PyTorch `flash_mla_sparse_fwd_sm120` on sm_120. Same signature.

KV is a DENSE [pool, D] bf16 buffer (already gathered+dequantized by
_forward_prefill); each query attends to `topk_length[q]` entries at `indices[q]`.
TRUE softmax with an attention sink added to the denominator only.

Design: one program per (query, head-block). Each query gathers its own sparse KV
(per-query indices => KV can't be shared across a query tile), then runs a
flash-style online softmax with tl.dot, tiling over the head dim H (the GEMM
M-dim) so the [BLOCK_H, D] accumulator fits in shared memory (D=512 is large):
  QK: q[BLOCK_H,D] . kv_tile.T[D,BLOCK_K] -> [BLOCK_H, BLOCK_K]   (bf16 MMA)
  AV: p[BLOCK_H,BLOCK_K] . kv_tile[BLOCK_K,D] -> [BLOCK_H, D]     (bf16 MMA)
Fuses QK->mask->softmax->sink->AV in one pass, reading each gathered KV tile
once (the PyTorch oracle reads it twice across two einsums and materializes the
full [Tq, topk, D] f32). attn_sink: e_sink = exp(sink - rowmax) in the denom only.
"""
import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634


@triton.jit
def _flash_mla_sparse_prefill_kernel(
    q_ptr, kv_ptr, idx_ptr, tl_ptr, sink_ptr, out_ptr,
    topk, sm_scale,
    D: tl.constexpr, H: tl.constexpr,
    TOPK: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_SINK: tl.constexpr, SINK_PER_H: tl.constexpr, LOG2E: tl.constexpr,
    stride_q_t, stride_q_h,
    stride_kv_p,
    stride_idx_t, stride_idx_k,
    stride_out_t, stride_out_h,
):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)             # [BLOCK_H]
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H

    q = tl.load(q_ptr + t * stride_q_t + offs_h[:, None] * stride_q_h
                + offs_d[None, :], mask=h_mask[:, None], other=0.0)   # [BLOCK_H, D] bf16

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)

    topk_len = tl.load(tl_ptr + t)
    scale_log2 = sm_scale * LOG2E

    for k0 in tl.range(0, topk, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)                   # [BLOCK_K]
        k_in = offs_k < topk
        in_len = offs_k < topk_len                            # [BLOCK_K]
        kv_idx = tl.load(idx_ptr + t * stride_idx_t + offs_k * stride_idx_k,
                         mask=k_in, other=0)                  # [BLOCK_K]
        kv_g = tl.load(kv_ptr + kv_idx[:, None] * stride_kv_p + offs_d[None, :],
                       mask=in_len[:, None], other=0.0)       # [BLOCK_K, D] bf16

        # QK [BLOCK_H, BLOCK_K] (bf16 MMA) -> scale + mask.
        qk = tl.dot(q, tl.trans(kv_g), out_dtype=tl.float32) * scale_log2
        qk = tl.where(in_len[None, :], qk, float("-inf"))

        # online softmax (exp2) over BLOCK_K, per head row.
        rowmax = tl.max(qk, axis=1)                           # [BLOCK_H]
        m_new = tl.maximum(m_i, rowmax)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        p = tl.where(in_len[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # AV [BLOCK_H, D] (bf16 MMA).
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv_g, out_dtype=tl.float32)
        m_i = m_new

    if HAS_SINK:
        if SINK_PER_H:
            sink_val = tl.load(sink_ptr + offs_h).to(tl.float32)          # [BLOCK_H]
        else:
            sink_val = tl.full([BLOCK_H], tl.load(sink_ptr).to(tl.float32),
                               dtype=tl.float32)
        e_sink = tl.math.exp2(sink_val * LOG2E - m_i)                     # [BLOCK_H]
        l_i = l_i + e_sink

    out = acc / tl.maximum(l_i, 1e-20)[:, None]                           # [BLOCK_H, D]
    tl.store(out_ptr + t * stride_out_t + offs_h[:, None] * stride_out_h
             + offs_d[None, :], out.to(tl.bfloat16), mask=h_mask[:, None])


def flash_mla_sparse_fwd_sm120_triton(
    q,            # [Tq, H, D] bf16
    kv,           # [pool, 1, D] bf16 (dense, already dequantized)
    indices,      # [Tq, 1, topk] int (index into the `pool` dim)
    sm_scale: float,
    attn_sink,    # scalar-tensor / [H] / None
    topk_length,  # [Tq] int
    out,          # [Tq, H, D] bf16 (written in place)
):
    """Write gathered sparse MLA attention into `out` (same contract as oracle)."""
    Tq, H, D = q.shape
    pool = kv.shape[0]
    kv_flat = kv.reshape(pool, D).contiguous()                  # [pool, D] bf16
    idx = indices.reshape(Tq, -1).contiguous()                  # [Tq, topk]
    topk = idx.shape[1]
    tl_flat = topk_length.reshape(Tq).to(torch.int32).contiguous()

    has_sink = attn_sink is not None
    sink_per_h = has_sink and attn_sink.dim() > 0
    sink_ptr = attn_sink.contiguous() if has_sink else q  # dummy ptr when unused

    BLOCK_H = 16
    grid = (Tq, triton.cdiv(H, BLOCK_H))
    _flash_mla_sparse_prefill_kernel[grid](
        q, kv_flat, idx, tl_flat, sink_ptr, out,
        topk, sm_scale,
        D=D, H=H, TOPK=triton.next_power_of_2(topk), BLOCK_H=BLOCK_H, BLOCK_K=32,
        HAS_SINK=has_sink, SINK_PER_H=sink_per_h, LOG2E=_LOG2E,
        stride_q_t=q.stride(0), stride_q_h=q.stride(1),
        stride_kv_p=kv_flat.stride(0),
        stride_idx_t=idx.stride(0), stride_idx_k=idx.stride(1),
        stride_out_t=out.stride(0), stride_out_h=out.stride(1),
        num_warps=4, num_stages=1,
    )
    return out
