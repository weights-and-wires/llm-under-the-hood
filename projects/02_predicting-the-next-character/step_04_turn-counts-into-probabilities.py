"""
Project 2: Step 4 — Turn counts into probabilities

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

P = N.float()
P = P / P.sum(dim=1, keepdim=True)

P = (N + 1).float()
P = P / P.sum(dim=1, keepdim=True)
