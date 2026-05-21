"""Unit tests for Project 5: tiny GPT from a blank file."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_build_module():
    spec = importlib.util.spec_from_file_location("project_05_build", PROJECT_DIR / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["project_05_build"] = module
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


class TestTokenizer:
    def test_roundtrip_on_corpus(self) -> None:
        stoi, itos, _ = build.char_tokenizer(build.DEFAULT_CORPUS)
        sample = build.DEFAULT_CORPUS[:200]
        encoded = build.encode(sample, stoi)
        decoded = build.decode(encoded, itos)
        assert decoded == sample

    def test_vocab_size_reasonable(self) -> None:
        _, _, vocab_size = build.char_tokenizer(build.DEFAULT_CORPUS)
        # Should be a few dozen characters, not hundreds or single digits.
        assert 30 < vocab_size < 100


class TestBatching:
    def test_batch_shapes(self) -> None:
        stoi, _, _ = build.char_tokenizer(build.DEFAULT_CORPUS)
        data = build.encode(build.DEFAULT_CORPUS, stoi)
        g = torch.Generator().manual_seed(0)
        x, y = build.get_batch(data, block_size=32, batch_size=8, generator=g)
        assert x.shape == (8, 32)
        assert y.shape == (8, 32)

    def test_targets_are_inputs_shifted_by_one(self) -> None:
        stoi, _, _ = build.char_tokenizer(build.DEFAULT_CORPUS)
        data = build.encode(build.DEFAULT_CORPUS, stoi)
        g = torch.Generator().manual_seed(0)
        # batch_size=1 makes the shift relationship visible.
        x, y = build.get_batch(data, block_size=32, batch_size=1, generator=g)
        # x[0, 1:] should equal y[0, :-1]
        assert torch.equal(x[0, 1:], y[0, :-1])


class TestModelArchitecture:
    def test_forward_shapes(self) -> None:
        cfg = build.GPTConfig(block_size=32, n_layers=2, n_heads=4, d_model=64)
        model = build.GPT(cfg, vocab_size=50)
        idx = torch.randint(0, 50, (2, 16))
        logits, loss = model(idx)
        assert logits.shape == (2, 16, 50)
        assert loss is None

    def test_forward_with_targets_returns_loss(self) -> None:
        torch.manual_seed(0)
        cfg = build.GPTConfig(block_size=32, n_layers=2, n_heads=4, d_model=64)
        model = build.GPT(cfg, vocab_size=50)
        idx = torch.randint(0, 50, (2, 16))
        targets = torch.randint(0, 50, (2, 16))
        logits, loss = model(idx, targets)
        assert logits.shape == (2, 16, 50)
        assert loss is not None
        assert torch.isfinite(loss).item()
        # Loss at init depends on the tied-embedding lm_head's variance; just
        # verify it's positive and finite. The training-converges test confirms
        # it actually drops.
        assert loss.item() > 0

    def test_weight_tying(self) -> None:
        cfg = build.GPTConfig(block_size=32, n_layers=2, n_heads=4, d_model=64)
        model = build.GPT(cfg, vocab_size=50)
        # lm_head.weight should be the same Parameter object as token_embedding.weight.
        assert model.lm_head.weight is model.token_embedding.weight

    def test_causal_mask_buffer_registered(self) -> None:
        cfg = build.GPTConfig(block_size=16, n_layers=1, n_heads=2, d_model=32)
        model = build.GPT(cfg, vocab_size=20)
        attn = model.blocks[0].attn
        # mask should be lower-triangular ones inside the (16,16) frame.
        assert attn.mask.shape == (1, 1, 16, 16)
        for i in range(16):
            for j in range(16):
                expected = 1.0 if j <= i else 0.0
                assert attn.mask[0, 0, i, j].item() == expected


class TestGeneration:
    def test_generate_extends_sequence(self) -> None:
        cfg = build.GPTConfig(block_size=32, n_layers=1, n_heads=2, d_model=32)
        model = build.GPT(cfg, vocab_size=20)
        model.eval()
        prompt = torch.tensor([[0, 1, 2]], dtype=torch.long)
        out = model.generate(prompt, max_new_tokens=10, temperature=1.0)
        assert out.shape == (1, 13)
        # The first 3 tokens must be preserved.
        assert torch.equal(out[0, :3], prompt[0])

    def test_generate_respects_block_size(self) -> None:
        """Generation should still work when current context exceeds block_size."""
        cfg = build.GPTConfig(block_size=8, n_layers=1, n_heads=2, d_model=32)
        model = build.GPT(cfg, vocab_size=20)
        model.eval()
        long_prompt = torch.randint(0, 20, (1, 20))  # longer than block_size
        out = model.generate(long_prompt, max_new_tokens=5, temperature=1.0)
        assert out.shape == (1, 25)


class TestTrainingConverges:
    def test_loss_decreases_substantially(self) -> None:
        """Tier-2 test: train briefly and confirm loss meaningfully decreases."""
        torch.manual_seed(0)
        stoi, _, vocab_size = build.char_tokenizer(build.DEFAULT_CORPUS)
        data = build.encode(build.DEFAULT_CORPUS, stoi)
        train_data = data[: int(0.9 * len(data))]
        val_data = data[int(0.9 * len(data)) :]
        cfg = build.GPTConfig(block_size=32, n_layers=2, n_heads=4, d_model=64)
        model = build.GPT(cfg, vocab_size=vocab_size)
        train_hist, _ = build.train_gpt(
            model, train_data, val_data, steps=100, batch_size=32, lr=3e-3, eval_every=20, seed=0
        )
        # Initial loss ~log(vocab), final should be at least 0.5 below.
        assert train_hist[0] > 3.0
        assert train_hist[-1] < train_hist[0] - 0.5
