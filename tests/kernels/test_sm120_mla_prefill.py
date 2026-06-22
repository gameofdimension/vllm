# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the sm_120 MLA prefill gathered-sparse attention (Op3).

`flash_mla_sparse_fwd_sm120_triton` (a flash-style online-softmax kernel with
the attention sink in the denominator) is checked against the original PyTorch
reference below (`ref_flash_mla_sparse_fwd`). The reference lived in the
production tree originally; once serving became Triton-only it was moved here as
a correctness oracle.

Outputs are bf16; the internal math is f32 in both, differing only in exp vs
exp2 / accumulation order. Measured max abs err ~6e-5; atol=1e-4 (rtol=0) is a
~1.7x margin.
"""
import pytest
import torch

from vllm.models.deepseek_v4.nvidia.flash_mla_sparse_prefill_triton import (
    flash_mla_sparse_fwd_sm120_triton,
)

H_MLA, D_MLA = 64, 512
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MLA_ATOL, MLA_RTOL = 1e-4, 0.0  # measured abs err ~6e-5 on bf16 output; rtol=0 -> pure atol


# --------------------------------------------------------------------------- #
# PyTorch reference (oracle) — verbatim from the former production module
# --------------------------------------------------------------------------- #
def ref_flash_mla_sparse_fwd(q, kv, indices, sm_scale, attn_sink, topk_length, out):
    """[Tq, H, D] bf16 gathered sparse MLA attention (prefill)."""
    Tq, H, D = q.shape
    kv_pool = kv.reshape(-1, D)
    idx = indices.reshape(Tq, -1)
    topk = idx.shape[1]
    idx_safe = idx.clamp(min=0).long()
    kv_g = kv_pool[idx_safe].to(torch.float32)
    qf = q.to(torch.float32) * sm_scale
    scores = torch.einsum("thd,tvd->thv", qf, kv_g)
    pos = torch.arange(topk, device=q.device)
    valid = (pos.unsqueeze(0) < topk_length.unsqueeze(1)).unsqueeze(1)
    scores = scores.masked_fill(~valid, float("-inf"))
    scores_max = scores.amax(dim=-1, keepdim=True)
    e = torch.exp(scores - scores_max)
    numer = torch.einsum("thv,tvd->thd", e, kv_g)
    denom = e.sum(dim=-1, keepdim=True)
    if attn_sink is not None:
        sink = attn_sink
        sink_b = sink.reshape(1, 1, 1) if sink.dim() == 0 else sink.reshape(1, -1, 1)
        denom = denom + torch.exp(sink_b.to(torch.float32) - scores_max)
    o = numer / denom.clamp(min=1e-20)
    out.copy_(o.to(out.dtype))
    return out


# --------------------------------------------------------------------------- #
# input generator
# --------------------------------------------------------------------------- #
def _make_inputs(Tq, topk, H=H_MLA, D=D_MLA):
    pool = topk * 2
    q = torch.randn(Tq, H, D, device=DEV, dtype=torch.bfloat16) * 0.05
    kv = torch.randn(pool, 1, D, device=DEV, dtype=torch.bfloat16) * 0.05
    indices = torch.randint(0, pool, (Tq, 1, topk), device=DEV, dtype=torch.int32)
    topk_length = torch.full((Tq,), topk, device=DEV, dtype=torch.int32)
    sm_scale = 1.0 / (D ** 0.5)
    return q, kv, indices, sm_scale, topk_length


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("Tq,topk", [(64, 512), (128, 2048)])
@pytest.mark.parametrize("sink", [
    pytest.param(None, id="no_sink"),
    pytest.param("scalar", id="scalar_sink"),
    pytest.param("per_h", id="per_h_sink"),
])
def test_mla_prefill(Tq, topk, sink):
    q, kv, indices, sm_scale, topk_length = _make_inputs(Tq, topk)
    sink_t = {
        None: None,
        "scalar": torch.tensor(-1.0, device=DEV),
        "per_h": torch.randn(H_MLA, device=DEV) * 0.1,
    }[sink]
    out_ref = torch.empty(Tq, H_MLA, D_MLA, device=DEV, dtype=torch.bfloat16)
    out_tri = torch.empty(Tq, H_MLA, D_MLA, device=DEV, dtype=torch.bfloat16)
    ref = ref_flash_mla_sparse_fwd(q, kv, indices, sm_scale, sink_t, topk_length, out_ref)
    tri = flash_mla_sparse_fwd_sm120_triton(q, kv, indices, sm_scale, sink_t, topk_length, out_tri)
    torch.testing.assert_close(tri, ref, rtol=MLA_RTOL, atol=MLA_ATOL)
