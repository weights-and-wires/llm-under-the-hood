"""
Project 11: Step 1 — Add per-layer gradient norm logging

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def group_grad_norms(model):
    norms = {}
    for name, module in model.named_modules():
        params = [p for p in module.parameters(recurse=False)
                  if p.grad is not None]
        if not params:
            continue
        total_sq = sum((p.grad ** 2).sum().item() for p in params)
        norms[name] = total_sq ** 0.5
    return norms
