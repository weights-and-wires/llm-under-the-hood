"""
Unit tests for Project 1's autograd engine and tiny MLP.

These are tier-2 tests per the repo convention: deterministic seed, tiny shapes,
target <30s. They verify the forward/backward semantics, not the wall-clock
training behavior (that's what build.py --tiny demonstrates by hand).
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_build_module():
    """Load build.py as a module without requiring the project to be a package."""
    spec = importlib.util.spec_from_file_location("project_01_build", PROJECT_DIR / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["project_01_build"] = module
    spec.loader.exec_module(module)
    return module


build = _load_build_module()
Value = build.Value
Neuron = build.Neuron
Layer = build.Layer
MLP = build.MLP


class TestValueForwardArithmetic:
    def test_add(self) -> None:
        a = Value(2.0)
        b = Value(3.0)
        assert (a + b).data == pytest.approx(5.0)

    def test_mul(self) -> None:
        a = Value(2.0)
        b = Value(3.0)
        assert (a * b).data == pytest.approx(6.0)

    def test_neg_and_sub(self) -> None:
        a = Value(5.0)
        b = Value(3.0)
        assert (-a).data == pytest.approx(-5.0)
        assert (a - b).data == pytest.approx(2.0)

    def test_pow(self) -> None:
        a = Value(2.0)
        assert (a**3).data == pytest.approx(8.0)

    def test_tanh_against_math(self) -> None:
        for x in [-2.0, -0.5, 0.0, 0.5, 2.0]:
            assert Value(x).tanh().data == pytest.approx(math.tanh(x), abs=1e-9)

    def test_exp_against_math(self) -> None:
        for x in [-1.0, 0.0, 1.0, 2.5]:
            assert Value(x).exp().data == pytest.approx(math.exp(x), abs=1e-9)

    def test_relu(self) -> None:
        assert Value(2.0).relu().data == pytest.approx(2.0)
        assert Value(-2.0).relu().data == pytest.approx(0.0)
        assert Value(0.0).relu().data == pytest.approx(0.0)


class TestValueGradients:
    def test_add_passes_gradient_straight_through(self) -> None:
        a, b = Value(2.0), Value(3.0)
        c = a + b
        c.backward()
        assert a.grad == pytest.approx(1.0)
        assert b.grad == pytest.approx(1.0)

    def test_mul_uses_partner_as_gradient(self) -> None:
        a, b = Value(2.0), Value(3.0)
        c = a * b
        c.backward()
        assert a.grad == pytest.approx(3.0)  # dc/da = b
        assert b.grad == pytest.approx(2.0)  # dc/db = a

    def test_tanh_gradient_matches_finite_difference(self) -> None:
        for x_val in [-1.5, -0.3, 0.0, 0.3, 1.5]:
            x = Value(x_val)
            y = x.tanh()
            y.backward()
            h = 1e-6
            fd = (math.tanh(x_val + h) - math.tanh(x_val - h)) / (2 * h)
            assert x.grad == pytest.approx(fd, rel=1e-4, abs=1e-6)

    def test_gradient_accumulates_over_multiple_paths(self) -> None:
        """If `a` is used twice, gradient should accumulate (not overwrite)."""
        a = Value(2.0)
        b = a + a  # b = 2a, db/da should be 2
        b.backward()
        assert a.grad == pytest.approx(2.0)

    def test_chain_rule_through_a_small_expression(self) -> None:
        """f(x) = (x * 2) + 3, df/dx should be 2."""
        x = Value(4.0)
        y = x * Value(2.0) + Value(3.0)
        y.backward()
        assert x.grad == pytest.approx(2.0)


class TestNeuronAndMLP:
    def test_neuron_call_returns_value_in_tanh_range(self) -> None:
        random.seed(0)
        n = Neuron(nin=3)
        out = n([1.0, 2.0, 3.0])
        assert isinstance(out, Value)
        assert -1.0 <= out.data <= 1.0

    def test_neuron_parameter_count(self) -> None:
        n = Neuron(nin=4)
        # 4 weights + 1 bias
        assert len(n.parameters()) == 5

    def test_mlp_parameter_count(self) -> None:
        random.seed(0)
        # nin=2 -> [3, 1]: layer1 has 3 neurons * (2 weights + 1 bias) = 9
        #                  layer2 has 1 neuron  * (3 weights + 1 bias) = 4
        m = MLP(nin=2, nouts=[3, 1])
        assert len(m.parameters()) == 13

    def test_mlp_forward_returns_single_value_for_1_output_neuron(self) -> None:
        random.seed(0)
        m = MLP(nin=2, nouts=[3, 1])
        out = m([0.5, -0.5])
        assert isinstance(out, Value)


class TestTrainingConverges:
    """End-to-end sanity check: 100 epochs on the toy dataset should converge."""

    def test_loss_decreases_substantially_in_100_epochs(self) -> None:
        random.seed(0)
        xs, ys = build.toy_dataset()
        model = MLP(nin=2, nouts=[3, 1])
        history = build.train(model, xs, ys, epochs=100, lr=0.05)
        # Initial loss should be O(1); final should be at least 10× smaller.
        assert history[0] > 0.5
        assert history[-1] < history[0] / 10

    def test_all_predictions_classify_correctly_after_training(self) -> None:
        random.seed(0)
        xs, ys = build.toy_dataset()
        model = MLP(nin=2, nouts=[3, 1])
        build.train(model, xs, ys, epochs=100, lr=0.05)
        preds = [model(x) for x in xs]
        for p, y in zip(preds, ys):
            # Sign of prediction should match sign of target.
            assert (p.data > 0) == (y > 0), f"misclassified: pred={p.data}, target={y}"
