"""
Project 11: Step 7 — Mixed-precision overflow detection

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

scaler = torch.cuda.amp.GradScaler()
overflow_count = 0
step_count = 0

# Inside the training loop:
scaler.scale(loss).backward()
prev_scale = scaler.get_scale()
scaler.step(optimizer)
scaler.update()
if scaler.get_scale() < prev_scale:
    overflow_count += 1
step_count += 1
