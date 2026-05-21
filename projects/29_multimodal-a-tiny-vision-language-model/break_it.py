"""
Project 29: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

projection = Projection(vision_dim=192, text_dim=384)
for p in projection.parameters():
    p.requires_grad = False
