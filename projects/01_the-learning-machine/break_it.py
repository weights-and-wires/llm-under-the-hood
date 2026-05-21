"""
Project 1: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out

def _backward():
    pass

def _backward():
    # self.grad += other.data * out.grad
    # other.grad += self.data * out.grad
    pass

for i, p in enumerate(model.parameters()[:10]):
    print(i, p.data, p.grad)

def _backward():
    self.grad += other.data * out.grad
    # other.grad += self.data * out.grad
