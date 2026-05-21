"""
Project 1: Step 1 — Build a Number That Remembers Where It Came From

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

x = 2.0
y = 3.0
z = x * y
print(z)  # 6.0

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
