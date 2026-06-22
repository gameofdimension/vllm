#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Standalone LATENCY bench for the sm_120 MLA decode Triton kernel
(_tiled_sparse_decode_kernel — 88.8% of decode GPU time per the §7.5 profile).

Values are random bytes (latency only, not correctness — correctness is covered
by tests/kernels + e2e 345). Cache layout = DSv4 fp8_ds_mla page:
  per page: [page_size*576 data | page_size*8 ue8m0 scale], stride(0)=page_size*584.
  per token data: [0:448] fp8 nope | [448:576] bf16 rope(64).
"""
import statistics
import sys

import torch

from vllm.models.deepseek_v4.nvidia.flash_mla_sm120_triton import (
    flash_mla_sparse_decode_triton,
)

DEV = torch.device("cuda:0")
PAGE = 64
TOPK = 512
D = 512
SM = 1.0 / (D ** 0.5)


def make_inputs(B, H, page_size=PAGE, topk=TOPK):
    num_pages = topk + 4  # enough real pages for the random indices
    # [num_pages, page_size, 1, 584] float8 ; stride(0) = page_size*584
    k = torch.randint(0, 256, (num_pages, page_size, 1, 584),
                      device=DEV, dtype=torch.uint8).view(torch.float8_e4m3fn)
    q = torch.randn(B, 1, H, D, device=DEV, dtype=torch.bfloat16) * 0.05
    indices = torch.randint(0, num_pages * page_size, (B, topk),
                            device=DEV, dtype=torch.int32)
    topk_length = torch.full((B,), topk, device=DEV, dtype=torch.int32)
    return q, k, indices, topk_length


def time_fn(args, warmup=8, iters=30):
    torch.cuda.synchronize(DEV)
    for _ in range(warmup):
        flash_mla_sparse_decode_triton(*args, None, D, SM, None, None, None)
    torch.cuda.synchronize(DEV)
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(iters)]
    for s, e in ev:
        s.record()
        flash_mla_sparse_decode_triton(*args, None, D, SM, None, None, None)
        e.record()
    torch.cuda.synchronize(DEV)
    return statistics.median(s.elapsed_time(e) for s, e in ev)


def main():
    torch.cuda.set_device(DEV)
    shapes = [(int(x.split(",")[0]), int(x.split(",")[1]))
              for x in sys.argv[1:]] or [(32, 64), (1, 64)]
    print(f"MLA decode kernel latency (topk={TOPK}, page={PAGE}, D={D})")
    for B, H in shapes:
        args = make_inputs(B, H)
        t = time_fn(args)
        print(f"  B={B:<3} H={H:<3}  {t:8.4f} ms/call  (grid={B*H} programs)")


if __name__ == "__main__":
    main()
