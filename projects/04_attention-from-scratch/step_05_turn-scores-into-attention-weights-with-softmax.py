"""
Project 4: Step 5 — Turn scores into attention weights with softmax

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

weights = torch.softmax(scores, dim=-1)
