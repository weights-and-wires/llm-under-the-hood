"""Unit tests for Project 4: attention from scratch."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_build_module():
    spec = importlib.util.spec_from_file_location("project_04_build", PROJECT_DIR / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["project_04_build"] = module
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


class TestCausalMask:
    def test_shape(self) -> None:
        m = build.causal_mask(5)
        assert m.shape == (5, 5)

    def test_lower_triangular(self) -> None:
        m = build.causal_mask(4)
        # Above the diagonal should be 0; diagonal and below should be 1.
        for i in range(4):
            for j in range(4):
                expected = 1.0 if j <= i else 0.0
                assert m[i, j].item() == expected


class TestSingleHeadAttention:
    def test_output_shape(self) -> None:
        T, d = 6, 16
        x = build.make_input_embeddings(T, d, seed=0)
        mask = build.causal_mask(T)
        W_Q, W_K, W_V, _ = build.init_projections(d, 1, seed=0)
        out, weights = build.single_head_attention(x, W_Q, W_K, W_V, mask, scale=True)
        assert out.shape == (T, d)
        assert weights.shape == (T, T)

    def test_weights_rows_sum_to_one(self) -> None:
        T, d = 6, 16
        x = build.make_input_embeddings(T, d, seed=0)
        mask = build.causal_mask(T)
        W_Q, W_K, W_V, _ = build.init_projections(d, 1, seed=0)
        _, weights = build.single_head_attention(x, W_Q, W_K, W_V, mask, scale=True)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_causal_zero_above_diagonal(self) -> None:
        T, d = 6, 16
        x = build.make_input_embeddings(T, d, seed=0)
        mask = build.causal_mask(T)
        W_Q, W_K, W_V, _ = build.init_projections(d, 1, seed=0)
        _, weights = build.single_head_attention(x, W_Q, W_K, W_V, mask, scale=True)
        for i in range(T):
            for j in range(i + 1, T):
                assert weights[i, j].item() == 0.0


class TestMultiHeadAttention:
    def test_shapes(self) -> None:
        T, d, H = 8, 32, 4
        x = build.make_input_embeddings(T, d, seed=0)
        mask = build.causal_mask(T)
        W_Q, W_K, W_V, W_O = build.init_projections(d, H, seed=0)
        out, w = build.multi_head_attention(x, W_Q, W_K, W_V, W_O, H, mask, scale=True)
        assert out.shape == (T, d)
        assert w.shape == (H, T, T)

    def test_each_head_row_sums_to_one(self) -> None:
        T, d, H = 8, 32, 4
        x = build.make_input_embeddings(T, d, seed=0)
        mask = build.causal_mask(T)
        W_Q, W_K, W_V, W_O = build.init_projections(d, H, seed=0)
        _, w = build.multi_head_attention(x, W_Q, W_K, W_V, W_O, H, mask, scale=True)
        row_sums = w.sum(dim=-1)  # (H, T)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


class TestAttentionEntropy:
    def test_uniform_distribution_maximizes_entropy(self) -> None:
        T = 5
        uniform = torch.ones(1, T, T) / T
        # Causal mask would zero some entries; use the unmasked uniform.
        ent = build.attention_entropy(uniform)
        expected = math.log(T)
        assert torch.allclose(ent, torch.full_like(ent, expected), atol=1e-5), (
            f"entropy of uniform should be log(T) = {expected}"
        )

    def test_one_hot_has_zero_entropy(self) -> None:
        one_hot = torch.zeros(1, 1, 5)
        one_hot[0, 0, 2] = 1.0
        ent = build.attention_entropy(one_hot)
        assert ent.item() == 0.0


class TestScalingMatters:
    """Verify the BREAK IT pedagogy: unscaled entropy is lower than scaled."""

    def test_scaled_has_higher_entropy_than_unscaled(self) -> None:
        T, d, H = 8, 64, 4
        x = build.make_input_embeddings(T, d, seed=0)
        mask = build.causal_mask(T)
        W_Q, W_K, W_V, W_O = build.init_projections(d, H, seed=0)
        _, w_scaled = build.multi_head_attention(x, W_Q, W_K, W_V, W_O, H, mask, scale=True)
        _, w_unscaled = build.multi_head_attention(x, W_Q, W_K, W_V, W_O, H, mask, scale=False)
        ent_scaled = build.attention_entropy(w_scaled).mean()
        ent_unscaled = build.attention_entropy(w_unscaled).mean()
        assert ent_scaled.item() > ent_unscaled.item(), (
            f"scaling should keep distributions less concentrated: "
            f"scaled={ent_scaled.item():.4f} vs unscaled={ent_unscaled.item():.4f}"
        )
