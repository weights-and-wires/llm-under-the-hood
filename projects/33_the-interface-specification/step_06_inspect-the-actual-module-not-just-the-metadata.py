"""
Project 33: Step 6 — Inspect the actual module, not just the metadata

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import torch

def check_norm_type(module: torch.nn.Module, expected: str) -> None:
    actual = type(module.norm).__name__
    if actual != expected:
        raise ValueError(f"Norm mismatch: expected {expected}, found {actual}")

def check_d_model(module: torch.nn.Module, expected: int) -> None:
    actual = module.attn.q_proj.weight.shape[1]
    if actual != expected:
        raise ValueError(f"d_model mismatch: expected {expected}, found {actual}")

def check_vocab_size(module: torch.nn.Module, expected: int) -> None:
    actual = module.lm_head.weight.shape[0]
    if actual != expected:
        raise ValueError(f"Vocab mismatch: expected {expected}, found {actual}")
