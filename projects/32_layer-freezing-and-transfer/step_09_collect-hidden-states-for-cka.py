"""
Project 32: Step 9 — Collect hidden states for CKA

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

with torch.no_grad():
    out_base = base_model(input_ids, output_hidden_states=True)
    out_ft = ft_model(input_ids, output_hidden_states=True)

base_hiddens = [h.reshape(-1, h.size(-1)) for h in out_base.hidden_states]
ft_hiddens = [h.reshape(-1, h.size(-1)) for h in out_ft.hidden_states]

import torch

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    dot_xy = Y.T @ X
    dot_xx = X.T @ X
    dot_yy = Y.T @ Y

    num = (dot_xy ** 2).sum()
    den = torch.sqrt((dot_xx ** 2).sum() * (dot_yy ** 2).sum())
    return (num / den).item()
