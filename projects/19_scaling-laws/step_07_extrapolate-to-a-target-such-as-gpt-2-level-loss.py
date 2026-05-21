"""
Project 19: Step 7 — Extrapolate to a target, such as GPT-2-level loss

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

L_target = 2.85  # example only
P_needed = (a / L_target) ** (1 / b)
print("Predicted parameters:", int(P_needed))
