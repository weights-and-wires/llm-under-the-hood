"""
Project 27: Step 2 — Implement basic INT8 post-training quantization

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import torch

def quantize_int8_symmetric(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = w.abs().max()
    scale = max_abs / 127.0 if max_abs > 0 else torch.tensor(1.0, device=w.device)
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale

def dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale
