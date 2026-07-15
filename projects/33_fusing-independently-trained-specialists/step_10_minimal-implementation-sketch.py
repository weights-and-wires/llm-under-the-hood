"""
Project 33: Step 10 — Minimal implementation sketch

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int = 3) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, n_experts)

    def forward(self, shared_h: torch.Tensor) -> torch.Tensor:
        # shared_h: [B, T, d_model]
        # use the last token as the routing summary
        x = shared_h[:, -1, :]              # [B, d_model]
        return F.softmax(self.proj(x), dim=-1)  # [B, 3]

import torch

def fuse_logits(
    router_weights: torch.Tensor, logits_list: list[torch.Tensor]
) -> torch.Tensor:
    # router_weights: [B, 3]
    # logits_list: three tensors of shape [B, T, V]
    fused = 0
    for i, logits in enumerate(logits_list):
        w = router_weights[:, i].view(-1, 1, 1)
        fused = fused + w * logits
    return fused
