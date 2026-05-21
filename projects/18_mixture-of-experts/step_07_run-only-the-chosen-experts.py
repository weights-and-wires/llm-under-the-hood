"""
Project 18: Step 7 — Run only the chosen experts

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
# shape: (B, T, N, d)

mask = torch.zeros_like(router_probs)
mask.scatter_(-1, topk_idx, topk_probs)
# shape: (B, T, N)

y = (expert_outputs * mask.unsqueeze(-1)).sum(dim=2)
# shape: (B, T, d)
