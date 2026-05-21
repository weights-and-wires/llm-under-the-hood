"""
Project 1: Step 12 — Run the Training Loop

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

for epoch in range(100):
    loss, preds = loss_fn(model, xs, ys)

    for p in model.parameters():
        p.grad = 0.0

    loss.backward()

    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    print(epoch, loss.data)
