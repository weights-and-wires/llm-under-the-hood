"""
Project 1: Step 7 — Verify the Gradients Numerically

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def finite_diff(f: callable, x: float, h: float = 1e-6) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)

x = Value(0.7)
y = x.tanh()
y.backward()
print("autograd:", x.grad)

import math
fd = (math.tanh(0.7 + 1e-6) - math.tanh(0.7 - 1e-6)) / (2e-6)
print("finite diff:", fd)
