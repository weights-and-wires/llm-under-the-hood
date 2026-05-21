"""
Project 4: Step 4 — Scale the scores

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

scores = (Q @ K.T) / math.sqrt(d_head)
