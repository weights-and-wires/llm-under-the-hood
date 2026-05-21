"""
Project 11: Step 3 — Add weight-update ratio

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def update_ratios(model, optimizer_step):
    pre_norms = {}
    pre_params = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            pre_norms[name] = p.data.norm().item()
            pre_params[name] = p.data.clone()
    optimizer_step()
    ratios = {}
    for name, p in model.named_parameters():
        if p.requires_grad and pre_norms[name] > 0:
            delta = (p.data - pre_params[name]).norm().item()
            ratios[name] = delta / pre_norms[name]
    return ratios
