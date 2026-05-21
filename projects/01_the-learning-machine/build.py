"""
Project 1: The Learning Machine — complete working build.

A scalar autograd engine plus a tiny MLP trained on a 2D toy dataset.
This is the assembled, runnable form of the code presented step-by-step
in Chapter 1 of *Under the Hood*.

The step_*.py files in this folder show each piece in isolation, the way
the book introduces them. This file is what they assemble into.

Run:
    python build.py --tiny              # 100 epochs, CPU, <5s
    python build.py --full              # 1000 epochs, more thorough
    python build.py --output-dir out    # write loss curve + final state to out/
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


class Value:
    """A scalar that remembers how it was produced, so we can walk gradients back."""

    def __init__(self, data: float, _children: tuple = (), _op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: float) -> "Value":
        assert isinstance(other, (int, float)), "only int/float exponents supported"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: "Value | float") -> "Value":
        return self + (-other if isinstance(other, Value) else -float(other))

    def __truediv__(self, other: "Value | float") -> "Value":
        if isinstance(other, Value):
            return self * (other ** -1)
        return self * (1.0 / float(other))

    def __radd__(self, other: float) -> "Value":
        return self + other

    def __rmul__(self, other: float) -> "Value":
        return self * other

    def __rsub__(self, other: float) -> "Value":
        return Value(other) - self

    def tanh(self) -> "Value":
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(0.0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Topologically order the graph, then run each node's local _backward in reverse."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self, nin: int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x: list[Value | float]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: list[Value | float]) -> Value | list[Value]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin: int, nouts: list[int]) -> None:
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

    def __call__(self, x: list[Value | float]) -> Value | list[Value]:
        out: Value | list[Value | float] = list(x)
        for layer in self.layers:
            out = layer(out if isinstance(out, list) else [out])
        return out

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


def toy_dataset() -> tuple[list[list[float]], list[float]]:
    """Two clusters in 2D — top-right is +1, bottom-left is -1."""
    xs = [
        [-1.2, -0.8],
        [-1.0, -1.1],
        [-0.8, -1.3],
        [1.0, 0.9],
        [1.2, 1.1],
        [0.8, 1.3],
    ]
    ys = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    return xs, ys


def squared_loss(model: MLP, xs: list[list[float]], ys: list[float]) -> Value:
    preds = [model(x) for x in xs]
    # preds is list[Value] because the final layer has 1 neuron.
    loss = sum(((p - y) ** 2 for p, y in zip(preds, ys)), Value(0.0))
    return loss


def train(
    model: MLP,
    xs: list[list[float]],
    ys: list[float],
    epochs: int,
    lr: float,
) -> list[float]:
    history: list[float] = []
    for epoch in range(epochs):
        loss = squared_loss(model, xs, ys)
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
        history.append(loss.data)
    return history


def write_loss_curve_png(history: list[float], path: Path) -> None:
    """Write the training loss curve as a PNG. Skips silently if matplotlib unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("squared loss")
    ax.set_title("Project 1: training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_run_log(
    history: list[float], preds_final: list[Value], ys: list[float], path: Path
) -> None:
    lines = [
        f"# Project 1 run log",
        f"final loss   : {history[-1]:.6f}",
        f"epoch 0 loss : {history[0]:.6f}",
        "",
        "# predictions vs targets",
    ]
    for p, y in zip(preds_final, ys):
        lines.append(f"  pred={p.data:+.4f}   target={y:+.1f}   |diff|={abs(p.data - y):.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiny", action="store_true", help="Proxy run (100 epochs).")
    parser.add_argument("--full", action="store_true", help="Full run (1000 epochs).")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
        help="Where to write loss curve + run log.",
    )
    args = parser.parse_args()

    if args.epochs is None:
        args.epochs = 1000 if args.full else 100

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    xs, ys = toy_dataset()
    model = MLP(nin=2, nouts=[3, 1])

    history = train(model, xs, ys, epochs=args.epochs, lr=args.lr)

    preds_final = [model(x) for x in xs]
    print(f"Project 1: trained {args.epochs} epochs (seed {args.seed}, lr {args.lr})")
    print(f"  initial loss: {history[0]:.6f}")
    print(f"  final loss  : {history[-1]:.6f}")
    print(f"  predictions:")
    for p, y in zip(preds_final, ys):
        sign = "OK" if (p.data > 0) == (y > 0) else "WRONG"
        print(f"    pred={p.data:+.4f}  target={y:+.1f}  {sign}")

    write_loss_curve_png(history, args.output_dir / "loss_curve.png")
    write_run_log(history, preds_final, ys, args.output_dir / "run_log.txt")
    print(f"\nOutputs written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
