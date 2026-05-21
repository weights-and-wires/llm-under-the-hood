"""
Project 27: Step 3 — Quantize one layer first, not the whole model

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import torch

def layer_error(linear: torch.nn.Linear, x: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        y_fp = linear(x)

        q_w, scale = quantize_int8_symmetric(linear.weight.data)
        w_hat = dequantize_int8(q_w, scale)

        y_q = x @ w_hat.t()
        if linear.bias is not None:
            y_q = y_q + linear.bias

        mse = torch.mean((y_fp - y_q) ** 2).item()
        max_err = torch.max((y_fp - y_q).abs()).item()
        return mse, max_err
