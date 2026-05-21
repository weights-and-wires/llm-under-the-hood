"""
Project 4: Step 14 — Measure per-head attention entropy

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

entropy = -(weights * (weights.clamp_min(1e-9).log())).sum(dim=-1)  # (H, T)
mean_entropy_per_head = entropy.mean(dim=-1)                        # (H,)
