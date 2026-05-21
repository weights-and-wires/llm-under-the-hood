"""Unit tests for Project 7: RMSNorm + SwiGLU alternatives."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_build_module():
    spec = importlib.util.spec_from_file_location("project_07_build", PROJECT_DIR / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["project_07_build"] = module
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


class TestRMSNorm:
    def test_output_shape(self) -> None:
        norm = build.RMSNorm(64)
        x = torch.randn(2, 4, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_unit_rms_after_normalization(self) -> None:
        """After RMSNorm with weight=ones, RMS over the last dim should be ~1."""
        norm = build.RMSNorm(64)
        x = torch.randn(8, 64) * 5.0
        out = norm(x)
        rms = torch.sqrt(out.pow(2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_weight_scales_output(self) -> None:
        norm = build.RMSNorm(8)
        with torch.no_grad():
            norm.weight.fill_(3.0)
        x = torch.randn(4, 8)
        out = norm(x)
        # Output should have RMS ~3 since weight is uniform 3.0.
        rms = torch.sqrt(out.pow(2).mean(dim=-1))
        assert torch.allclose(rms, torch.full_like(rms, 3.0), atol=1e-3)


class TestSwiGLU:
    def test_output_shape(self) -> None:
        mlp = build.SwiGLU(64)
        x = torch.randn(2, 4, 64)
        out = mlp(x)
        assert out.shape == (2, 4, 64)

    def test_no_bias_in_swiglu_linears(self) -> None:
        mlp = build.SwiGLU(64)
        assert mlp.gate_proj.bias is None
        assert mlp.up_proj.bias is None
        assert mlp.down_proj.bias is None


class TestModernGPT:
    def test_forward_with_rmsnorm_and_swiglu(self) -> None:
        cfg = build.p5.GPTConfig(block_size=32, n_layers=2, n_heads=4, d_model=64)
        model = build.ModernGPT(cfg, vocab_size=50, norm_type="rms", mlp_type="swiglu")
        idx = torch.randint(0, 50, (2, 16))
        targets = torch.randint(0, 50, (2, 16))
        logits, loss = model(idx, targets)
        assert logits.shape == (2, 16, 50)
        assert loss is not None
        assert torch.isfinite(loss).item()

    def test_all_four_variants_trainable(self) -> None:
        text = build.p5.DEFAULT_CORPUS
        stoi, _, V = build.p5.char_tokenizer(text)
        data = build.p5.encode(text, stoi)
        n_train = int(0.9 * len(data))
        for norm_type in ["ln", "rms"]:
            for mlp_type in ["gelu", "swiglu"]:
                _, train_loss, val_loss = build.train_variant(
                    norm_type,
                    mlp_type,
                    V,
                    data[:n_train],
                    data[n_train:],
                    steps=50,
                    seed=0,
                )
                # All variants should at least reduce loss below uniform.
                assert train_loss < 3.5, (
                    f"{norm_type}+{mlp_type} did not train: train_loss={train_loss}"
                )
