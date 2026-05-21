"""Unit tests for Project 2: bigram + neural character LM."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_build_module():
    spec = importlib.util.spec_from_file_location("project_02_build", PROJECT_DIR / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["project_02_build"] = module
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


class TestVocab:
    def test_dot_token_is_index_zero(self) -> None:
        _, stoi, itos = build.build_vocab(["abc", "def"])
        assert stoi["."] == 0
        assert itos[0] == "."

    def test_vocab_contains_all_characters(self) -> None:
        chars, stoi, _ = build.build_vocab(["abc", "def"])
        for ch in "abcdef.":
            assert ch in stoi
        assert len(chars) == 7  # 6 letters + '.'

    def test_stoi_and_itos_are_inverses(self) -> None:
        _, stoi, itos = build.build_vocab(["xyz", "abc"])
        for ch, idx in stoi.items():
            assert itos[idx] == ch


class TestBigramModel:
    def test_count_matrix_shape(self) -> None:
        _, stoi, _ = build.build_vocab(["abc"])
        V = len(stoi)
        N = build.build_bigram_counts(["abc"], stoi)
        assert N.shape == (V, V)
        assert N.dtype == torch.int32

    def test_specific_transitions_counted(self) -> None:
        _, stoi, _ = build.build_vocab(["ab"])
        N = build.build_bigram_counts(["ab"], stoi)
        # ".ab." should produce (., a), (a, b), (b, .)
        assert N[stoi["."], stoi["a"]] == 1
        assert N[stoi["a"], stoi["b"]] == 1
        assert N[stoi["b"], stoi["."]] == 1

    def test_probs_rows_sum_to_one(self) -> None:
        _, stoi, _ = build.build_vocab(["abcdef"])
        N = build.build_bigram_counts(["abcdef"], stoi)
        P = build.bigram_probs(N, smoothing=1)
        row_sums = P.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_smoothing_eliminates_zero_probabilities(self) -> None:
        _, stoi, _ = build.build_vocab(["ab"])
        N = build.build_bigram_counts(["ab"], stoi)
        # without smoothing, most cells are 0; with smoothing, none are.
        P_smoothed = build.bigram_probs(N, smoothing=1)
        assert (P_smoothed > 0).all().item()

    def test_nll_is_positive(self) -> None:
        _, stoi, _ = build.build_vocab(build.DEFAULT_NAMES)
        N = build.build_bigram_counts(build.DEFAULT_NAMES, stoi)
        P = build.bigram_probs(N, smoothing=1)
        nll = build.bigram_nll(P, build.DEFAULT_NAMES, stoi)
        assert nll > 0
        # For 24-char vocab, uniform NLL would be log(24) ~ 3.18.
        # A useful model should beat uniform.
        assert nll < math.log(24)


class TestNeuralTrainingData:
    def test_shape_for_one_word(self) -> None:
        _, stoi, _ = build.build_vocab(["abc"])
        X, Y = build.build_neural_training_data(["abc"], stoi, block_size=3)
        # word "abc" with block_size 3 gives 4 examples: (...->a), (..a->b), (.ab->c), (abc->.)
        assert X.shape == (4, 3)
        assert Y.shape == (4,)

    def test_first_example_is_all_dot(self) -> None:
        _, stoi, _ = build.build_vocab(["abc"])
        X, _ = build.build_neural_training_data(["abc"], stoi, block_size=3)
        assert X[0].tolist() == [0, 0, 0]  # three '.' tokens

    def test_last_target_is_end_token(self) -> None:
        _, stoi, _ = build.build_vocab(["abc"])
        _, Y = build.build_neural_training_data(["abc"], stoi, block_size=3)
        assert Y[-1].item() == 0  # '.' (end)


class TestNeuralCharLM:
    def test_parameter_shapes(self) -> None:
        m = build.NeuralCharLM(vocab_size=10, block_size=3, embed_dim=4, hidden_size=8, seed=0)
        assert m.C.shape == (10, 4)
        assert m.W1.shape == (12, 8)  # block_size * embed_dim, hidden_size
        assert m.b1.shape == (8,)
        assert m.W2.shape == (8, 10)
        assert m.b2.shape == (10,)

    def test_forward_output_shape(self) -> None:
        m = build.NeuralCharLM(vocab_size=10, block_size=3, embed_dim=4, hidden_size=8, seed=0)
        X = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
        logits = m.forward(X)
        assert logits.shape == (2, 10)

    def test_loss_is_finite_at_init(self) -> None:
        m = build.NeuralCharLM(vocab_size=10, block_size=3, embed_dim=4, hidden_size=8, seed=0)
        X = torch.tensor([[0, 1, 2]], dtype=torch.long)
        Y = torch.tensor([3])
        loss = m.loss(X, Y)
        assert torch.isfinite(loss).item()
        assert loss.item() > 0


class TestTrainingConverges:
    """End-to-end: 500 steps on the default names corpus should beat random uniform."""

    @pytest.mark.parametrize("steps", [200])
    def test_train_loss_drops_substantially(self, steps: int) -> None:
        torch.manual_seed(0)
        words = build.DEFAULT_NAMES
        _, stoi, _ = build.build_vocab(words)
        V = len(stoi)
        X, Y = build.build_neural_training_data(words, stoi, block_size=3)
        n_train = int(X.shape[0] * 0.85)
        model = build.NeuralCharLM(vocab_size=V, block_size=3, embed_dim=8, hidden_size=64, seed=0)
        train_hist, _ = build.train_neural(
            model,
            X[:n_train],
            Y[:n_train],
            X[n_train:],
            Y[n_train:],
            steps=steps,
            batch_size=32,
            lr=0.1,
            eval_every=max(1, steps // 5),
            seed=0,
        )
        # log(V) ~ 3.18 for V=24; initial random loss is around there. Final
        # train loss should be well below that.
        assert train_hist[-1] < math.log(V) - 0.3

    def test_neural_model_beats_uniform_on_val(self) -> None:
        torch.manual_seed(0)
        words = build.DEFAULT_NAMES
        _, stoi, _ = build.build_vocab(words)
        V = len(stoi)
        X, Y = build.build_neural_training_data(words, stoi, block_size=3)
        n_train = int(X.shape[0] * 0.85)
        model = build.NeuralCharLM(vocab_size=V, block_size=3, embed_dim=8, hidden_size=64, seed=0)
        _, val_hist = build.train_neural(
            model,
            X[:n_train],
            Y[:n_train],
            X[n_train:],
            Y[n_train:],
            steps=300,
            batch_size=32,
            lr=0.1,
            eval_every=50,
            seed=0,
        )
        assert val_hist[-1] < math.log(V)  # better than uniform


class TestSampling:
    def test_bigram_samples_are_strings(self) -> None:
        _, stoi, itos = build.build_vocab(build.DEFAULT_NAMES)
        N = build.build_bigram_counts(build.DEFAULT_NAMES, stoi)
        P = build.bigram_probs(N, smoothing=1)
        samples = build.bigram_sample(P, itos, n_samples=5, seed=0)
        assert len(samples) == 5
        assert all(isinstance(s, str) for s in samples)

    def test_neural_sampling_returns_strings_no_dot(self) -> None:
        torch.manual_seed(0)
        _, stoi, itos = build.build_vocab(build.DEFAULT_NAMES)
        V = len(stoi)
        model = build.NeuralCharLM(vocab_size=V, block_size=3, embed_dim=8, hidden_size=64, seed=0)
        samples = model.sample(itos, stoi, n_samples=5, temperature=1.0, seed=0)
        assert len(samples) == 5
        # samples should not contain the '.' end-of-sequence token
        assert all("." not in s for s in samples)
