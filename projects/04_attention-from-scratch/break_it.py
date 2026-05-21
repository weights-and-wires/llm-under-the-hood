"""
Project 4: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

scores = (Q @ K.T) / math.sqrt(d_head)

scores = Q @ K.T

scores = scores.masked_fill(mask == 0, float('-inf'))
