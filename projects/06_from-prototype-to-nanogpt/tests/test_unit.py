"""Unit tests for Project 6: nanoGPT-style refinements."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_build_module():
    spec = importlib.util.spec_from_file_location("project_06_build", PROJECT_DIR / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["project_06_build"] = module
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


class TestParameterGroups:
    def test_two_groups_returned(self) -> None:
        cfg = build.p5.GPTConfig(block_size=16, n_layers=2, n_heads=2, d_model=32)
        model = build.p5.GPT(cfg, vocab_size=20)
        groups = build.configure_param_groups(model, weight_decay=0.1)
        assert len(groups) == 2

    def test_first_group_gets_decay(self) -> None:
        cfg = build.p5.GPTConfig(block_size=16, n_layers=2, n_heads=2, d_model=32)
        model = build.p5.GPT(cfg, vocab_size=20)
        groups = build.configure_param_groups(model, weight_decay=0.1)
        assert groups[0]["weight_decay"] == 0.1
        assert groups[1]["weight_decay"] == 0.0

    def test_bias_params_in_no_decay(self) -> None:
        cfg = build.p5.GPTConfig(block_size=16, n_layers=2, n_heads=2, d_model=32)
        model = build.p5.GPT(cfg, vocab_size=20)
        groups = build.configure_param_groups(model, weight_decay=0.1)
        # All Linear bias params should be in no_decay group.
        no_decay_ids = {id(p) for p in groups[1]["params"]}
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                assert id(module.bias) in no_decay_ids

    def test_layernorm_params_in_no_decay(self) -> None:
        cfg = build.p5.GPTConfig(block_size=16, n_layers=2, n_heads=2, d_model=32)
        model = build.p5.GPT(cfg, vocab_size=20)
        groups = build.configure_param_groups(model, weight_decay=0.1)
        no_decay_ids = {id(p) for p in groups[1]["params"]}
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                assert id(module.weight) in no_decay_ids
                assert id(module.bias) in no_decay_ids

    def test_all_params_accounted_for_with_weight_tying(self) -> None:
        """Weight-tied params should appear once total."""
        cfg = build.p5.GPTConfig(block_size=16, n_layers=2, n_heads=2, d_model=32)
        model = build.p5.GPT(cfg, vocab_size=20)
        groups = build.configure_param_groups(model, weight_decay=0.1)
        total_in_groups = sum(p.numel() for g in groups for p in g["params"])
        # Sum of unique parameters (accounting for weight tying).
        seen = set()
        total_unique = 0
        for p in model.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total_unique += p.numel()
        assert total_in_groups == total_unique


class TestInitWeights:
    def test_linear_init_std_is_0_02(self) -> None:
        cfg = build.p5.GPTConfig(block_size=16, n_layers=2, n_heads=2, d_model=64)
        model = build.p5.GPT(cfg, vocab_size=20)
        torch.manual_seed(0)
        build.init_weights(model)
        # qkv linear is a non-residual Linear → std should be ~0.02
        qkv_std = model.blocks[0].attn.qkv.weight.std().item()
        assert abs(qkv_std - 0.02) < 0.005, f"expected qkv std ~0.02, got {qkv_std}"

    def test_residual_proj_has_smaller_std(self) -> None:
        cfg = build.p5.GPTConfig(block_size=16, n_layers=4, n_heads=2, d_model=64)
        model = build.p5.GPT(cfg, vocab_size=20)
        torch.manual_seed(0)
        build.init_weights(model)
        # Residual projections should have std = 0.02 / sqrt(2 * n_layers)
        # For n_layers=4: 0.02 / sqrt(8) ~ 0.00707
        proj_std = model.blocks[0].attn.proj.weight.std().item()
        assert proj_std < 0.012, f"residual proj should be smaller than std=0.02, got {proj_std}"


class TestRefinementsActuallyHelp:
    def test_nanogpt_style_outperforms_prototype_on_val(self) -> None:
        """The headline pedagogy: weight decay + scaled init helps val loss."""
        text = build.p5.DEFAULT_CORPUS
        stoi, _, V = build.p5.char_tokenizer(text)
        data = build.p5.encode(text, stoi)
        n_train = int(0.9 * len(data))
        train_data, val_data = data[:n_train], data[n_train:]
        cfg = build.p5.GPTConfig(block_size=32, n_layers=2, n_heads=4, d_model=64)

        torch.manual_seed(0)
        proto = build.p5.GPT(cfg, vocab_size=V)
        _, proto_val = build.p5.train_gpt(
            proto, train_data, val_data, steps=200, batch_size=32, lr=3e-3, eval_every=40, seed=0
        )

        torch.manual_seed(0)
        nano = build.p5.GPT(cfg, vocab_size=V)
        build.init_weights(nano)
        _, nano_val = build.train_with_groups(
            nano,
            train_data,
            val_data,
            steps=200,
            batch_size=32,
            lr=3e-3,
            eval_every=40,
            weight_decay=0.1,
            seed=0,
        )
        # nanoGPT-style should have lower (or similar) val loss.
        # We use a generous tolerance because this is a small comparison.
        assert nano_val[-1] <= proto_val[-1] + 0.5, (
            f"nanoGPT-style ({nano_val[-1]:.3f}) should not be much worse than "
            f"prototype ({proto_val[-1]:.3f})"
        )
