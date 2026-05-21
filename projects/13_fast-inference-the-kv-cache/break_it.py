"""
Project 13: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

# Example: corrupt layer 3, head 0, position 57
with torch.no_grad():
    cache.k[3][:, 0, 57, :] = torch.randn_like(cache.k[3][:, 0, 57, :])
