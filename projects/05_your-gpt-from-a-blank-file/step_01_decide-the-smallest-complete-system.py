"""
Project 5: Step 1 — Decide the smallest complete system

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

from dataclasses import dataclass

@dataclass
class Config:
    batch_size = 64
    block_size = 128
    d_model = 256
    n_heads = 4
    n_layers = 6
    dropout = 0.2
    learning_rate = 3e-4
    min_lr = 3e-5
    warmup_steps = 200
    max_steps = 10000
    eval_interval = 200
    eval_steps = 50
    grad_clip = 1.0
    device = "cuda"
