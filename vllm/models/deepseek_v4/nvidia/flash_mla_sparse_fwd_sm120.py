# SPDX-License-Identifier: Apache-2.0
# sm_120 replacement for vLLM's flash_mla_sparse_fwd in the DeepSeek-V4 prefill
# path. The original is a FlashMLA C kernel (SM90a/SM100f only). Here the KV has
# already been gathered+dequantized into a DENSE bf16 buffer by _forward_prefill,
# so we only need gathered sparse attention over that dense buffer — expressible
# in plain PyTorch. See sm120/CHANGES.md.
"""sm_120 gathered sparse MLA attention for the DeepSeek-V4 PREFILL path.

Replaces flash_mla_sparse_fwd(q, kv=dense bf16, indices, topk_length, attn_sink,
out). KV is a dense [pool, 1, D] bf16 buffer (gathered by _forward_prefill);
each query attends to topk_length[q] entries at indices[q].
"""
import torch


def flash_mla_sparse_fwd_sm120(
    q,            # [Tq, H, D] bf16
    kv,           # [pool, 1, D] bf16 (dense, already dequantized)
    indices,      # [Tq, 1, topk] int (index into the `pool` dim)
    sm_scale: float,
    attn_sink,    # scalar / [H] / None  (logit added to softmax denom)
    topk_length,  # [Tq] int (valid count per query)
    out,          # [Tq, H, D] bf16 (output buffer, written in place)
):
    Tq, H, D = q.shape
    kv_pool = kv.reshape(-1, D)                       # [pool, D]
    idx = indices.reshape(Tq, -1)                     # [Tq, topk]
    topk = idx.shape[1]

    idx_safe = idx.clamp(min=0).long()
    kv_g = kv_pool[idx_safe].to(torch.float32)        # [Tq, topk, D]
    qf = q.to(torch.float32) * sm_scale               # [Tq, H, D]

    scores = torch.einsum("thd,tvd->thv", qf, kv_g)   # [Tq, H, topk]

    # Mask out indices beyond each query's valid length.
    pos = torch.arange(topk, device=q.device)
    valid = (pos.unsqueeze(0) < topk_length.unsqueeze(1)).unsqueeze(1)  # [Tq,1,topk]
    scores = scores.masked_fill(~valid, float("-inf"))

    scores_max = scores.amax(dim=-1, keepdim=True)    # [Tq, H, 1]
    e = torch.exp(scores - scores_max)                # [Tq, H, topk] (0 where masked)
    numer = torch.einsum("thv,tvd->thd", e, kv_g)     # [Tq, H, D]
    denom = e.sum(dim=-1, keepdim=True)               # [Tq, H, 1]

    if attn_sink is not None:
        # Attention sink: a logit added to the softmax denominator (a sink token
        # that absorbs probability mass but contributes no output value).
        sink = attn_sink
        if sink.dim() == 0:
            sink_b = sink.reshape(1, 1, 1)
        else:
            sink_b = sink.reshape(1, -1, 1)           # [1, H, 1] (or broadcasts)
        e_sink = torch.exp(sink_b.to(torch.float32) - scores_max)
        denom = denom + e_sink

    o = numer / denom.clamp(min=1e-20)
    out.copy_(o.to(out.dtype))
    return out
