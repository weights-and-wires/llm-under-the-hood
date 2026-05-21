"""
Project 13: Step 1 — Start with naive autoregressive generation

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

@torch.no_grad()
def generate_naive(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
    # idx: [B, T]
    for _ in range(max_new_tokens):
        logits = model(idx)              # full forward pass on full sequence
        last = logits[:, -1, :] / temperature
        next_id = torch.argmax(last, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx

import time

import torch

def bench(fn, *args, new_tokens: int = 128, warmup: int = 1, runs: int = 3, **kwargs) -> float:
    for _ in range(warmup):
        _ = fn(*args, max_new_tokens=new_tokens, **kwargs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.perf_counter()
    for _ in range(runs):
        _ = fn(*args, max_new_tokens=new_tokens, **kwargs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt = time.perf_counter() - start
    return (runs * new_tokens) / dt
