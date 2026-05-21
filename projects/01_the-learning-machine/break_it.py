"""
Project 1: BREAK IT — kill the gradient flow through multiplication.

We replace `Value.__mul__`'s backward pass with a no-op. The forward pass
still works (predictions look normal at epoch 0), but no blame ever flows
back through any multiplication. Since every weight is multiplied by an
input somewhere, the gradients never reach the parameters — and the loss
refuses to fall.

Run:
    python break_it.py --tiny

Expected output: final loss is roughly the same as initial loss
(no training happened), and many predictions are WRONG.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from build import MLP, Value, squared_loss, toy_dataset


def install_broken_mul() -> None:
    """Sabotage: __mul__ forward is unchanged but _backward propagates nothing."""

    def __mul__(self: Value, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # SABOTAGE: gradient flow through multiplication is removed.
            # Forward still works; backward is a no-op.
            pass

        out._backward = _backward
        return out

    Value.__mul__ = __mul__  # type: ignore[method-assign]


def train_and_report(
    model: MLP,
    xs: list[list[float]],
    ys: list[float],
    epochs: int,
    lr: float,
    label: str,
) -> tuple[float, float]:
    initial_loss = squared_loss(model, xs, ys).data
    for _ in range(epochs):
        loss = squared_loss(model, xs, ys)
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
    final_loss = squared_loss(model, xs, ys).data
    print(f"{label:20s}  initial={initial_loss:.6f}  final={final_loss:.6f}")
    return initial_loss, final_loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiny", action="store_true", help="Proxy run (100 epochs).")
    parser.add_argument("--full", action="store_true", help="Full run (1000 epochs).")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
    )
    args = parser.parse_args()

    if args.epochs is None:
        args.epochs = 1000 if args.full else 100

    xs, ys = toy_dataset()

    # First: working baseline.
    random.seed(args.seed)
    baseline = MLP(nin=2, nouts=[3, 1])
    base_initial, base_final = train_and_report(baseline, xs, ys, args.epochs, args.lr, "baseline")

    # Now: sabotage and re-run.
    install_broken_mul()
    random.seed(args.seed)
    broken = MLP(nin=2, nouts=[3, 1])
    broken_initial, broken_final = train_and_report(broken, xs, ys, args.epochs, args.lr, "broken __mul__")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = args.output_dir / "break_it_log.txt"
    log.write_text(
        f"# Project 1 BREAK IT — kill gradient flow through __mul__\n"
        f"baseline:        initial={base_initial:.6f}  final={base_final:.6f}\n"
        f"broken __mul__:  initial={broken_initial:.6f}  final={broken_final:.6f}\n"
        f"\n"
        f"Lesson: without gradient flow through multiplication, every weight is\n"
        f"effectively untrained. The forward pass produces output, but no learning\n"
        f"happens. The final loss is roughly the initial loss.\n",
        encoding="utf-8",
    )
    print(f"\nLog written to {log.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
