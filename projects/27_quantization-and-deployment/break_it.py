"""
Project 27: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

# INT4
scale = max_abs / 7.0
q = torch.clamp(torch.round(w / scale), -8, 7)

# INT2-ish
scale = max_abs / 1.0
q = torch.clamp(torch.round(w / scale), -2, 1)
