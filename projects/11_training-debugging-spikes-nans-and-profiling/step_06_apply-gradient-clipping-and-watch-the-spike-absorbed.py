"""
Project 11: Step 6 — Apply gradient clipping and watch the spike absorbed

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
