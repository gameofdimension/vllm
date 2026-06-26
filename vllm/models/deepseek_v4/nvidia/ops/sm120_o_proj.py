# SPDX-License-Identifier: Apache-2.0
# sm_120 o_proj helpers, isolated so the edit to ops/o_proj.py is a one-line
# re-export (keeps the patch surface tiny). See sm120/CHANGES.md.
"""DeepSeek-V4 attention o-projection for sm_120 (no DeepGEMM).

Mirrors SGLang's default o_proj (its fused-DeepGEMM version is an opt-in
optimization, off by default): inverse-RoPE in bf16 + a plain bf16 grouped
einsum over wo_a + wo_b. wo_a is FP8 e4m3 at runtime (the bf16 checkpoint
weight is online-quantized at load) with a float32 [128,128] block scale, so
it is dequantized to bf16 (cached) before the einsum.
"""
import torch
import torch.nn as nn


def _sm120_inv_rope(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """DeepSeek-V4 interleaved (GPT-J style) INVERSE RoPE in bf16 (pure torch).

    Mirrors the inverse-RoPE math of the Triton kernel `fused_inv_rope_fp8_quant`
    minus the FP8 quant. Only the last `rope_dim` elements of each head rotate.
    `cos_sin_cache` is `[max_pos, rope_dim]` = cos(0..R/2-1) || sin(0..R/2-1).
    """
    R = rope_dim
    half = R // 2
    T, H, _ = o.shape
    rope = o[..., -R:].to(torch.float32)
    cs = cos_sin_cache[positions.long()]
    cos = cs[:, :half].unsqueeze(1)
    sin = cs[:, half:].unsqueeze(1)
    pairs = rope.reshape(T, H, half, 2)
    x_even = pairs[..., 0]
    x_odd = pairs[..., 1]
    out_even = x_even * cos + x_odd * sin
    out_odd = x_odd * cos - x_even * sin
    out = torch.stack((out_even, out_odd), dim=-1).reshape(T, H, R)
    o = o.clone()
    o[..., -R:] = out.to(o.dtype)
    return o


def _sm120_wo_a_bf16(wo_a: nn.Module) -> torch.Tensor:
    """Return wo_a.weight as bf16, dequantizing FP8 block-quant on the fly
    (cached on the module as ``wo_a._sm120_bf16``)."""
    cached = getattr(wo_a, "_sm120_bf16", None)
    if cached is not None:
        return cached
    w = wo_a.weight
    if w.dtype == torch.bfloat16:
        wo_a._sm120_bf16 = w
        return w
    wf = w.to(torch.float32)
    scale = getattr(wo_a, "weight_scale_inv", None)
    if scale is None:
        scale = getattr(wo_a, "weight_scale", None)
    bs = getattr(wo_a, "weight_block_size", [128, 128])
    # Our fp8 checkpoint stores wo_a as BF16; vLLM re-quantizes it to FP8 on
    # load but leaves a corrupt FLT_MAX placeholder scale (a properly quantized
    # fp8 weight spans ±448, but here w_absmax ~0.2 = the raw bf16 values). In
    # that case the fp8 values ARE the true weight values, so use them as-is.
    scale_corrupt = (
        scale is not None and scale.dtype == torch.float32
        and bool(torch.isfinite(scale).all())
        and float(scale.abs().max().item()) > 1.0e4
    )
    w_unquantized = float(wf.abs().max().item()) < 10.0  # proper fp8 spans ~448
    if scale_corrupt or w_unquantized:
        out = wf.to(torch.bfloat16)
        wo_a._sm120_bf16 = out
        return out
    if (
        scale is not None
        and scale.dtype == torch.float32
        and scale.ndim == 2
        and isinstance(bs, (list, tuple))
        and len(bs) == 2
    ):
        for s_try in (scale, scale.t()):
            s = (
                s_try.contiguous()
                .repeat_interleave(bs[0], dim=0)
                .repeat_interleave(bs[1], dim=1)
            )
            if s.shape == wf.shape:
                wf = wf * s
                break
    elif scale is not None and scale.numel() == 1:
        wf = wf * float(scale.reshape(-1)[0].item())
    out = wf.to(torch.bfloat16)
    wo_a._sm120_bf16 = out
    return out


def sm120_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    rope_dim: int,
    o_lora_rank: int,
):
    """sm_120 o-projection without DeepGEMM.

    inv-RoPE (bf16) -> reshape [T, n_groups, d] -> bf16 grouped einsum over
    wo_a (float32 accumulation) -> wo_b.
    """
    T = o.shape[0]
    o = _sm120_inv_rope(o, positions, cos_sin_cache, rope_dim)
    o = o.reshape(T, n_groups, -1)
    wo_a_bf16 = _sm120_wo_a_bf16(wo_a).view(n_groups, o_lora_rank, -1)
    o = torch.einsum(
        "tgd,grd->tgr", o.float(), wo_a_bf16.float()
    ).to(torch.bfloat16)
    out = wo_b(o.reshape(T, -1))
    if isinstance(out, tuple):
        out = out[0]
    return out
