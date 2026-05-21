"""
Project 11: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

lr = base_lr * 3.0
