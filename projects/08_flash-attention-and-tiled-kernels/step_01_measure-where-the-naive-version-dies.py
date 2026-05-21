"""
Project 8: Step 1 — Measure where the naive version dies

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def naive_attention(q, k, v):
    # q, k, v: (batch, heads, seq, dim)
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)   # (B, H, N, N)
    weights = torch.softmax(scores, dim=-1)            # (B, H, N, N)
    return weights @ v                                  # (B, H, N, D)

import torch

def measure_naive(seq_lens, batch=2, heads=8, dim=64, device="cuda", dtype=torch.float16):
    results = {}
    for n in seq_lens:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        q = torch.randn(batch, heads, n, dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        try:
            out = naive_attention(q, k, v)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e9
            results[n] = peak
        except torch.cuda.OutOfMemoryError:
            results[n] = None
        del q, k, v
        if results[n] is not None:
            del out
    return results

print(measure_naive([1024, 2048, 4096, 8192]))
