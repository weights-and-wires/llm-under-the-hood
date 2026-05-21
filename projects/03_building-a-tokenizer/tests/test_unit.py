"""Unit tests for Project 3: byte-level BPE tokenizer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_build_module():
    spec = importlib.util.spec_from_file_location("project_03_build", PROJECT_DIR / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["project_03_build"] = module
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


class TestGetStats:
    def test_counts_adjacent_pairs(self) -> None:
        ids = [1, 2, 3, 2, 3]
        counts = build.get_stats(ids)
        assert counts[(1, 2)] == 1
        assert counts[(2, 3)] == 2
        assert counts[(3, 2)] == 1

    def test_empty_list_yields_no_pairs(self) -> None:
        assert dict(build.get_stats([])) == {}
        assert dict(build.get_stats([7])) == {}


class TestMerge:
    def test_merges_matching_pair(self) -> None:
        assert build.merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]

    def test_does_not_overlap(self) -> None:
        # Overlapping pairs: [1,1,1] with pair (1,1) -> should produce [99, 1], not [99, 99].
        assert build.merge([1, 1, 1], (1, 1), 99) == [99, 1]

    def test_no_matches_returns_original(self) -> None:
        assert build.merge([1, 2, 3], (4, 5), 99) == [1, 2, 3]


class TestBPETrainBasic:
    def test_train_below_256_raises(self) -> None:
        tok = build.BPETokenizer()
        try:
            tok.train("hello", vocab_size=100)
        except ValueError:
            return
        raise AssertionError("expected ValueError for vocab_size < 256")

    def test_train_at_256_learns_zero_merges(self) -> None:
        tok = build.BPETokenizer()
        size = tok.train("hello world hello world", vocab_size=256)
        assert size == 256
        assert len(tok.merges) == 0

    def test_train_actually_learns_merges_for_repetitive_text(self) -> None:
        tok = build.BPETokenizer()
        # Heavy repetition forces real merges.
        text = "hello hello hello hello world world world"
        size = tok.train(text, vocab_size=300)
        assert size > 256
        assert len(tok.merges) > 0


class TestRoundtrip:
    def test_encode_decode_identity_on_ascii(self) -> None:
        tok = build.BPETokenizer()
        tok.train(build.DEFAULT_CORPUS, vocab_size=512)
        for s in build.SAMPLE_SENTENCES:
            assert tok.decode(tok.encode(s)) == s

    def test_encode_decode_identity_on_unicode(self) -> None:
        tok = build.BPETokenizer()
        tok.train(build.DEFAULT_CORPUS, vocab_size=512)
        for s in ["café", "naïve", "日本語"]:
            assert tok.decode(tok.encode(s)) == s

    def test_empty_string_roundtrips(self) -> None:
        tok = build.BPETokenizer()
        tok.train(build.DEFAULT_CORPUS, vocab_size=512)
        assert tok.decode(tok.encode("")) == ""


class TestCompressionTradeoff:
    def test_more_vocab_means_fewer_tokens(self) -> None:
        tok256 = build.BPETokenizer()
        tok256.train(build.DEFAULT_CORPUS, vocab_size=256)
        tok512 = build.BPETokenizer()
        tok512.train(build.DEFAULT_CORPUS, vocab_size=512)
        sentence = "The quick brown fox jumps over the lazy dog."
        # vocab=256 = bytes; vocab=512 should compress much more.
        assert len(tok512.encode(sentence)) < len(tok256.encode(sentence))

    def test_compression_ratio_increases_with_vocab(self) -> None:
        tok256 = build.BPETokenizer()
        tok256.train(build.DEFAULT_CORPUS, vocab_size=256)
        tok512 = build.BPETokenizer()
        tok512.train(build.DEFAULT_CORPUS, vocab_size=512)
        _, _, r256 = build.compression_ratio(build.DEFAULT_CORPUS, tok256)
        _, _, r512 = build.compression_ratio(build.DEFAULT_CORPUS, tok512)
        assert r256 == 1.0  # byte-level, no compression
        assert r512 > 2.0  # real compression


class TestEncodeMergePriority:
    def test_earlier_merge_wins_when_both_applicable(self) -> None:
        """If merges (a,b)->X and (X,c)->Y both exist, then `abc` should encode to [Y]."""
        tok = build.BPETokenizer()
        # Manually construct a tiny tokenizer.
        tok.merges = {}
        tok.vocab = {i: bytes([i]) for i in range(256)}
        a, b, c = ord("a"), ord("b"), ord("c")
        tok.merges[(a, b)] = 256
        tok.vocab[256] = bytes([a, b])
        tok.merges[(256, c)] = 257
        tok.vocab[257] = bytes([a, b, c])
        # 'abc' should collapse all the way to a single token via two merges.
        assert tok.encode("abc") == [257]
