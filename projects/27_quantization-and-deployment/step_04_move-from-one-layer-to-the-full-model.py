"""
Project 27: Step 4 — Move from one layer to the full model

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import torch

def quantize_model_int8(model: torch.nn.Module) -> dict:
    qstate = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            q_w, scale = quantize_int8_symmetric(module.weight.data)
            qstate[name] = {
                "q_weight": q_w.cpu(),
                "scale": scale.cpu(),
                "bias": None if module.bias is None else module.bias.data.cpu()
            }
    return qstate
