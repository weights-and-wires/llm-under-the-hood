"""
Project 27: Step 6 — Implement INT4 quantization

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import torch

def quantize_int4_groupwise(
    w: torch.Tensor, group_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = w.flatten()
    n = flat.numel()

    q_groups = []
    scales = []

    for i in range(0, n, group_size):
        chunk = flat[i:i+group_size]
        max_abs = chunk.abs().max()
        scale = max_abs / 7.0 if max_abs > 0 else torch.tensor(1.0, device=w.device)
        q = torch.clamp(torch.round(chunk / scale), -8, 7).to(torch.int8)
        q_groups.append(q)
        scales.append(scale)

    q = torch.cat(q_groups).view_as(w)
    scales = torch.stack(scales)
    return q, scales
