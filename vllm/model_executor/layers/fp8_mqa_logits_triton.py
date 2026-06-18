# SPDX-License-Identifier: Apache-2.0
"""sm_120 fused Triton scorers for the c4 MQA indexer (decode + prefill).

Two kernels, each a drop-in for its chunked-PyTorch oracle on sm_120 (behind
is_deep_gemm_supported()):

- `fp8_paged_mqa_logits_sm120_triton` — PAGED decode scorer. KV is the indexer
  cache [num_blocks, block_size, 1, D+4] uint8 == 128B fp8_e4m3fn value + 4B
  float32 scale per token (a per-token f32 scale, NOT the MLA decode kernel's
  ue8m0-per-group-of-64). Addresses tokens via block_tables. Output [Bn, max_model_len].
- `fp8_mqa_logits_sm120_triton` — DENSE prefill scorer. KV is already gathered
  into [N, D] fp8 + [N] f32 scale; per-query valid range [ks, ke). Output [M, N].

Score math (no softmax, no attn_sink — raw weighted scores feed downstream topk):
  for each KV token t:  score[t] = ( sum_h relu(q[h] . k[t]) * w[h] ) * k_scale[t]
  invalid positions -> -inf. qk uses fp8 tensor-core tl.dot (f32 accumulator).

Grid: one program per (kv-tile, query-row). No cross-tile reduction (no softmax).
"""
from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_mqa_logits_kernel(
    q_ptr, w_ptr, kv_fp8_ptr, kv_f32_ptr, bt_ptr, cl_ptr, logits_ptr,
    max_model_len, T_eff,
    block_size,
    D: tl.constexpr, REC: tl.constexpr,
    BLOCK_TOK: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr,
    stride_q_row, stride_q_h,
    stride_w_row, stride_w_h,
    stride_bt_row, stride_bt_page,
    stride_log_row,
):
    pid_tok = tl.program_id(0)
    row = tl.program_id(1)

    offs_tok = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)   # [BLOCK_TOK]
    offs_d = tl.arange(0, BLOCK_D)                              # [BLOCK_D]
    offs_h = tl.arange(0, BLOCK_H)                              # [BLOCK_H]
    tok_in_eff = offs_tok < T_eff                               # has a KV page

    ctx_len = tl.load(cl_ptr + row)
    valid = (offs_tok < ctx_len) & (offs_tok < max_model_len)   # within context & output

    # Paged address: token -> (page, slot) -> block id -> byte base of its record.
    page_id = offs_tok // block_size
    slot = offs_tok % block_size
    block_ids = tl.load(bt_ptr + row * stride_bt_row + page_id * stride_bt_page,
                        mask=tok_in_eff, other=0)               # [BLOCK_TOK]
    base = (block_ids.to(tl.int64) * (block_size * REC)
            + slot.to(tl.int64) * REC)                          # [BLOCK_TOK] byte offset

    # KV values [BLOCK_TOK, D] (kept fp8 for the tensor-core GEMM) + per-token f32 scale.
    val_off = base[:, None] + offs_d[None, :]                   # [BLOCK_TOK, BLOCK_D] bytes
    kv_val = tl.load(kv_fp8_ptr + val_off,
                     mask=tok_in_eff[:, None], other=0.0)        # fp8 [BLOCK_TOK, D]
    sc_idx = (base + D) // 4                                    # [BLOCK_TOK] f32-element index
    kv_sc = tl.load(kv_f32_ptr + sc_idx,
                    mask=tok_in_eff, other=0.0).to(tl.float32)   # [BLOCK_TOK]

    # qk [BLOCK_TOK, H] = kv_val . q^T   (fp8 tensor-core MMA, f32 accumulator)
    q_row = tl.load(q_ptr + row * stride_q_row
                    + offs_h[:, None] * stride_q_h + offs_d[None, :])   # [H, D] fp8
    qk = tl.dot(kv_val, tl.trans(q_row), out_dtype=tl.float32)          # [BLOCK_TOK, H]
    qk = tl.maximum(qk, 0.0)                                            # relu

    # score = ( sum_h relu(qk) * w_h ) * kv_sc
    w_vec = tl.load(w_ptr + row * stride_w_row + offs_h * stride_w_h)   # [H] f32
    score = tl.sum(qk * w_vec[None, :], axis=1) * kv_sc                 # [BLOCK_TOK]
    score = tl.where(valid, score, float("-inf"))

    tl.store(logits_ptr + row * stride_log_row + offs_tok, score,
             mask=offs_tok < max_model_len)


def fp8_paged_mqa_logits_sm120_triton(
    q,                  # tuple (q_values [B, next_n, H, D] fp8, q_scale None)
    kv_cache,           # [num_blocks, block_size, 1, D+4] uint8
    weights,            # [B*next_n, H] f32
    context_lens,       # [B, next_n] int32
    block_tables,       # [B, max_blocks] int32
    schedule_metadata: Any,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """Return logits [B*next_n, max_model_len] f32 (same contract as the oracle)."""
    _ = schedule_metadata  # DeepGEMM-only
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError(
            "sm_120 triton scorer: FP4 indexer q not supported; FP8 path only."
        )

    B, next_n, H, D = q_values.shape
    assert D == 128, "DSV4 indexer head_dim must be 128"
    num_blocks, block_size, _, hw = kv_cache.shape
    assert hw == D + 4, f"expected per-token width {D + 4}, got {hw}"
    Bn = B * next_n

    qf = q_values.reshape(Bn, H, D).contiguous()                 # [Bn, H, D] fp8
    w = weights.reshape(Bn, H).contiguous()                      # [Bn, H] f32
    sl = context_lens.reshape(-1).to(torch.int32).contiguous()   # [Bn]

    pt = block_tables
    if pt.shape[0] != Bn:
        pt = pt.unsqueeze(1).expand(-1, next_n, -1).reshape(Bn, -1)
    pt = pt.contiguous()
    max_pages = pt.shape[1]

    logits = q_values.new_empty((Bn, max_model_len), dtype=torch.float32)
    logits.fill_(float("-inf"))

    # Two typed views of one flat KV buffer: fp8 (1 B/elt) for values, f32 (4 B/elt) for scales.
    raw = kv_cache.view(torch.uint8).reshape(-1)
    kv_fp8 = raw.view(torch.float8_e4m3fn)
    kv_f32 = raw.view(torch.float32)

    REC = D + 4
    T_eff = min(max_model_len, max_pages * block_size)
    BLOCK_TOK = 64
    BLOCK_D = triton.next_power_of_2(D)
    grid = (triton.cdiv(T_eff, BLOCK_TOK), Bn)
    _paged_mqa_logits_kernel[grid](
        qf, w, kv_fp8, kv_f32, pt, sl, logits,
        max_model_len, T_eff,
        block_size,
        D=D, REC=REC,
        BLOCK_TOK=BLOCK_TOK, BLOCK_D=BLOCK_D, BLOCK_H=triton.next_power_of_2(H),
        stride_q_row=qf.stride(0), stride_q_h=qf.stride(1),
        stride_w_row=w.stride(0), stride_w_h=w.stride(1),
        stride_bt_row=pt.stride(0), stride_bt_page=pt.stride(1),
        stride_log_row=logits.stride(0),
        num_warps=4, num_stages=2,
    )
    return logits


# --------------------------------------------------------------------------- #
# DENSE (non-paged) prefill scorer — Op2
# --------------------------------------------------------------------------- #
@triton.jit
def _dense_mqa_logits_kernel(
    q_ptr, w_ptr, k_fp8_ptr, k_sc_ptr, ks_ptr, ke_ptr, logits_ptr,
    N,
    D: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr,
    stride_q_m, stride_q_h,
    stride_w_m, stride_w_h,
    stride_k_n, stride_k_d,
    stride_log_m,
):
    pid_k = tl.program_id(0)
    m = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)            # [BLOCK_K]
    offs_d = tl.arange(0, BLOCK_D)                               # [BLOCK_D]
    offs_h = tl.arange(0, BLOCK_H)                               # [BLOCK_H]
    k_in = offs_k < N

    ks = tl.load(ks_ptr + m)
    ke = tl.load(ke_ptr + m)
    valid = (offs_k >= ks) & (offs_k < ke) & k_in                # query's [ks, ke) range

    # Dense KV [BLOCK_K, D] fp8 + per-position f32 scale [BLOCK_K].
    k_off = offs_k[:, None] * stride_k_n + offs_d[None, :] * stride_k_d
    kv_val = tl.load(k_fp8_ptr + k_off, mask=k_in[:, None], other=0.0)   # fp8
    ksc = tl.load(k_sc_ptr + offs_k, mask=k_in, other=0.0).to(tl.float32)  # [BLOCK_K]

    # qk [BLOCK_K, H] = kv_val . q^T  (fp8 tensor-core MMA)
    q_row = tl.load(q_ptr + m * stride_q_m
                    + offs_h[:, None] * stride_q_h + offs_d[None, :])     # [H, D] fp8
    qk = tl.dot(kv_val, tl.trans(q_row), out_dtype=tl.float32)            # [BLOCK_K, H]
    qk = tl.maximum(qk, 0.0)

    w_vec = tl.load(w_ptr + m * stride_w_m + offs_h * stride_w_h)        # [H] f32
    score = tl.sum(qk * w_vec[None, :], axis=1) * ksc                     # [BLOCK_K]
    score = tl.where(valid, score, float("-inf"))

    tl.store(logits_ptr + m * stride_log_m + offs_k, score, mask=k_in)


def fp8_mqa_logits_sm120_triton(
    q,             # tuple (q_values [M, H, D] fp8, q_scale None)
    kv,            # tuple (k_packed [N, D] fp8, k_scales [N] f32)
    weights,       # [M, H] f32
    cu_seqlen_ks,  # [M] int32
    cu_seqlen_ke,  # [M] int32
    clean_logits: bool,
) -> torch.Tensor:
    """Return logits [M, N] f32 (same contract as the PyTorch oracle)."""
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError(
            "sm_120 triton prefill scorer: FP4 q not supported; FP8 path only."
        )
    k_packed, k_scales = kv
    M, H, D = q_values.shape
    assert D == 128, "DSV4 indexer head_dim must be 128"
    N = k_packed.shape[0]

    qf = q_values.contiguous()                                    # [M, H, D] fp8
    w = weights.contiguous()                                      # [M, H] f32
    ksc = k_scales.contiguous().view(torch.float32).reshape(N)    # [N] f32
    ks = cu_seqlen_ks.reshape(M).to(torch.int32).contiguous()
    ke = cu_seqlen_ke.reshape(M).to(torch.int32).contiguous()

    logits = q_values.new_empty((M, N), dtype=torch.float32)
    logits.fill_(float("-inf"))

    n_max = min(int(ke.max().item()), N) if N > 0 else 0          # only score up to max used K
    if n_max > 0:
        BLOCK_K = 64
        BLOCK_D = triton.next_power_of_2(D)
        grid = (triton.cdiv(n_max, BLOCK_K), M)
        _dense_mqa_logits_kernel[grid](
            qf, w, k_packed, ksc, ks, ke, logits,
            N,
            D=D, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, BLOCK_H=triton.next_power_of_2(H),
            stride_q_m=qf.stride(0), stride_q_h=qf.stride(1),
            stride_w_m=w.stride(0), stride_w_h=w.stride(1),
            stride_k_n=k_packed.stride(0), stride_k_d=k_packed.stride(1),
            stride_log_m=logits.stride(0),
            num_warps=4, num_stages=2,
        )
    return logits
