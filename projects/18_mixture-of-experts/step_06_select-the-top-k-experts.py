"""
Project 18: Step 6 — Select the top-k experts

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

topk_probs, topk_idx = torch.topk(router_probs, k=2, dim=-1)

topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
