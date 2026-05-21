"""
Project 32: Step 11 — Training target for the router

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

with torch.no_grad():
    losses = torch.stack([loss_code, loss_med, loss_legal], dim=-1)  # [B, 3]
    targets = F.softmax(-losses / temp, dim=-1)

weights = router(shared_h)  # [B, 3]
router_loss = -(targets * weights.log()).sum(dim=-1).mean()
