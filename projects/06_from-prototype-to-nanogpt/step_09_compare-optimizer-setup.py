"""
Project 6: Step 9 — Compare optimizer setup

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
