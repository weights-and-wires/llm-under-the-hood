"""
Project 2: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

with torch.no_grad():
    C[:] = C[0]
