"""
Project 36: Step 5 — Decide where the architecture change lives

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

# x: anchor or task batch
# base_model: frozen reference
# specialist: trainable model or adapter
# lambda_interface: strength of interface preservation

task_loss = specialist.task_loss(task_batch)

with torch.no_grad():
    base_h = base_model.forward(anchor_batch, return_hidden=True)

spec_h = specialist.forward(anchor_batch, return_hidden=True)

interface_loss = mse(spec_h[layer_id], base_h[layer_id])

loss = task_loss + lambda_interface * interface_loss
loss.backward()
