"""
Project 1: Step 6 — Add the Rest of the Small Operations

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def __neg__(self):
    return self * -1

def __sub__(self, other):
    return self + (-other)

def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
        self.grad += (other * self.data**(other - 1)) * out.grad

    out._backward = _backward
    return out

def __truediv__(self, other):
    return self * (other ** -1)

def exp(self):
    x = self.data
    out = Value(math.exp(x), (self,), 'exp')

    def _backward():
        self.grad += out.data * out.grad

    out._backward = _backward
    return out

def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    def _backward():
        self.grad += (out.data > 0) * out.grad

    out._backward = _backward
    return out
