"""
Project 5: Step 9 — Count parameters before and after weight tying

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def count_params(model):
    return sum(p.numel() for p in model.parameters())
