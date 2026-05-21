"""
Project 1: Step 4 — Add a Nonlinearity

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import math

def tanh(self) -> 'Value':
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
        self.grad += (1 - out.data**2) * out.grad

    out._backward = _backward
    return out
