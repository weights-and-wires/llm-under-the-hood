"""
Project 18: Step 9 — Track expert utilization

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

with torch.no_grad():
    flat_idx = topk_idx.reshape(-1)
    counts = torch.bincount(flat_idx, minlength=num_experts)
