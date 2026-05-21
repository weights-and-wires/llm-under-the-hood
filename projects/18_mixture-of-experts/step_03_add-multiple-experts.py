"""
Project 18: Step 3 — Add multiple experts

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

self.experts = nn.ModuleList([
    Expert(d_model, d_ff) for _ in range(num_experts)
])
