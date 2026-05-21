"""Unit tests for Project 8: tiled attention vs. naive attention."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_build_module():
    spec = importlib.util.spec_from_file_location("project_08_build", PROJECT_DIR / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["project_08_build"] = module
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


class TestNaiveAttention:
    def test_output_shape(self) -> None:
        Q, K, V = build.build_random_qkv(T=32, d_head=16, seed=0)
        out, _ = build.naive_attention(Q, K, V, causal=True)
        assert out.shape == (32, 16)

    def test_peak_memory_is_T_squared(self) -> None:
        T, d = 64, 16
        Q, K, V = build.build_random_qkv(T=T, d_head=d, seed=0)
        _, peak = build.naive_attention(Q, K, V, causal=True)
        # We allocate scores (T,T) plus weights (T,T).
        assert peak == 2 * T * T


class TestTiledAttention:
    def test_output_shape(self) -> None:
        Q, K, V = build.build_random_qkv(T=64, d_head=16, seed=0)
        out, _ = build.tiled_attention(Q, K, V, q_block=16, kv_block=16, causal=True)
        assert out.shape == (64, 16)

    @pytest.mark.parametrize(
        "T,d_head,qb,kb",
        [
            (16, 8, 8, 8),
            (32, 16, 16, 16),
            (64, 16, 16, 32),
            (128, 32, 32, 32),
            (97, 16, 16, 16),  # non-divisible T
            (128, 32, 17, 33),  # weird block sizes
        ],
    )
    def test_tiled_matches_naive(self, T: int, d_head: int, qb: int, kb: int) -> None:
        Q, K, V = build.build_random_qkv(T=T, d_head=d_head, seed=0)
        out_naive, _ = build.naive_attention(Q, K, V, causal=True)
        out_tiled, _ = build.tiled_attention(Q, K, V, q_block=qb, kv_block=kb, causal=True)
        max_diff = (out_naive - out_tiled).abs().max().item()
        assert max_diff < 1e-5, (
            f"tiled differed from naive by {max_diff:.2e} at T={T},d={d_head},qb={qb},kb={kb}"
        )

    def test_tiled_peak_memory_is_less_than_naive_for_large_T(self) -> None:
        T, d = 128, 32
        Q, K, V = build.build_random_qkv(T=T, d_head=d, seed=0)
        _, peak_naive = build.naive_attention(Q, K, V, causal=True)
        _, peak_tiled = build.tiled_attention(Q, K, V, q_block=32, kv_block=32, causal=True)
        assert peak_tiled < peak_naive, (
            f"tiled peak ({peak_tiled}) should be less than naive ({peak_naive})"
        )
