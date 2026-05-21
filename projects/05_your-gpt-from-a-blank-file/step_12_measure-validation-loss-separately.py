"""
Project 5: Step 12 — Measure validation loss separately

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

@torch.no_grad()
def estimate_loss(split: str) -> float:
    model.eval()
    losses = torch.zeros(cfg.eval_steps)
    for k in range(cfg.eval_steps):
        xb, yb = get_batch(split, cfg)
        _, loss = model(xb, yb)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()
