"""
Project 18: Step 11 — Add the load-balancing auxiliary loss

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

importance = router_probs.mean(dim=(0, 1))  # (N,)
load = mask.sum(dim=(0, 1)) / mask.sum()    # (N,)

aux_loss = num_experts * torch.sum(importance * load)

loss = lm_loss + alpha * aux_loss
