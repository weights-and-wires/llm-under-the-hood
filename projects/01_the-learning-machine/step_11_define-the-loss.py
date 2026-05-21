"""
Project 1: Step 11 — Define the Loss

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def loss_fn(model: 'MLP', xs: list, ys: list) -> tuple:
    preds = [model(x) for x in xs]
    loss = sum((pred - y)**2 for pred, y in zip(preds, ys))
    return loss, preds
