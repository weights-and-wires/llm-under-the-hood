"""
Project 32: Step 3 — Freeze parameters correctly

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import torch

def freeze_bottom_layers(model: torch.nn.Module, freeze_ratio: float) -> int:
    n_layers = len(model.transformer.h)
    freeze_upto = int(n_layers * freeze_ratio)

    for layer_idx, block in enumerate(model.transformer.h):
        freeze = layer_idx < freeze_upto
        for p in block.parameters():
            p.requires_grad = not freeze

    return freeze_upto

import logging

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
logging.info(f"trainable: {trainable}/{total} = {trainable/total:.2%}")
