"""
Project 3: Building a Tokenizer — complete working build.

A byte-level Byte-Pair-Encoding (BPE) tokenizer, trained from scratch on a small
built-in corpus of mixed English prose, code, and URLs. Trains multiple
vocabulary sizes and reports the compression vs. fragmentation tradeoff.

Run:
    python build.py --tiny             # vocab sizes [288, 512, 1024], CPU <30s
    python build.py --full             # vocab sizes [288, 512, 1024, 2048, 4096]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

# Built-in mixed corpus: prose + code + URLs + identifiers. ~3KB.
DEFAULT_CORPUS = """
The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.
She sells seashells by the seashore. She sells seashells by the seashore.
A language model predicts the next token given the previous tokens.
A language model predicts the next token given the previous tokens.
Compression is what tokenizers actually optimize for, not meaning.

def main():
    user_id = 42
    name = "alice"
    return user_id, name

def fetch(url: str):
    response = httpx.get(url)
    return response.json()

class Embedding:
    def __init__(self, vocab_size, dim):
        self.weight = torch.randn(vocab_size, dim)

    def forward(self, ids):
        return self.weight[ids]

SELECT id, name, created_at FROM users WHERE active = TRUE ORDER BY created_at DESC LIMIT 100;
SELECT id, name, email FROM users WHERE id = 42;

https://example.com/api/v1/users/42
https://example.com/api/v1/users/123
https://docs.python.org/3/library/argparse.html

The training loop iterates over batches.
The training loop iterates over batches.
loss.backward()
optimizer.step()
optimizer.zero_grad()

replaying the song, replaying the song
playing playing playing
internationalization, internationalization
the cat sat on the mat. the cat sat on the mat.

color colour, color colour
favorite favourite
"""


def get_stats(ids: list[int]) -> Counter[tuple[int, int]]:
    """Count adjacent pair occurrences across `ids`."""
    counts: Counter[tuple[int, int]] = Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE: starts from 256 byte tokens, learns merges greedily."""

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> int:
        """Train merges until the corpus has no more multi-occurrence pairs
        OR vocab_size is reached. Returns the actual final vocab size."""
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256 for byte-level BPE (got {vocab_size})")
        ids = list(text.encode("utf-8"))
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        n_merges = vocab_size - 256
        for k in range(n_merges):
            stats = get_stats(ids)
            if not stats or max(stats.values()) < 2:
                # No pair occurs more than once → further merges would just
                # invent one-off tokens. Stop early.
                break
            pair = max(stats, key=lambda p: stats[p])
            new_id = 256 + k
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (k < 5 or k % 100 == 0):
                piece = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"  merge {new_id}: {pair} (count={stats[pair]}) -> {piece!r}")
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            # Find the lowest-rank merge present among adjacent pairs.
            stats = {(a, b) for a, b in zip(ids, ids[1:])}
            best = None
            best_rank = float("inf")
            for pair in stats:
                if pair in self.merges and self.merges[pair] < best_rank:
                    best = pair
                    best_rank = self.merges[pair]
            if best is None:
                break
            ids = merge(ids, best, self.merges[best])
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")


def compression_ratio(text: str, tokenizer: BPETokenizer) -> tuple[int, int, float]:
    raw_bytes = len(text.encode("utf-8"))
    n_tokens = len(tokenizer.encode(text))
    return raw_bytes, n_tokens, raw_bytes / max(n_tokens, 1)


def token_frequency_stats(tokenizer: BPETokenizer, text: str) -> dict:
    ids = tokenizer.encode(text)
    counts = Counter(ids)
    n_tokens = len(counts)
    n_lt_5 = sum(1 for c in counts.values() if c < 5)
    n_eq_1 = sum(1 for c in counts.values() if c == 1)
    return {
        "unique_tokens_used": n_tokens,
        "median_freq": sorted(counts.values())[n_tokens // 2] if n_tokens else 0,
        "lt_5_frac": n_lt_5 / max(n_tokens, 1),
        "eq_1_frac": n_eq_1 / max(n_tokens, 1),
    }


def visualize_tokens(tokenizer: BPETokenizer, text: str) -> str:
    """Pretty-print a string as `[token][token][token]` boundaries."""
    ids = tokenizer.encode(text)
    pieces = []
    for tid in ids:
        s = tokenizer.vocab[tid].decode("utf-8", errors="replace")
        # Make whitespace visible
        s = s.replace(" ", "·").replace("\n", "↵")
        pieces.append(f"[{s}]")
    return "".join(pieces)


def write_compression_plot(results: list[tuple[int, float, float]], path: Path) -> None:
    """results: list of (vocab_size, avg_tokens_per_sample, compression_ratio)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    sizes = [r[0] for r in results]
    avg_tokens = [r[1] for r in results]
    ratios = [r[2] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(sizes, avg_tokens, "o-")
    ax1.set_xscale("log")
    ax1.set_xlabel("vocab size (log)")
    ax1.set_ylabel("avg tokens per sample sentence")
    ax1.set_title("Tokens per sentence vs vocab size")
    ax1.grid(True, alpha=0.3)
    ax2.plot(sizes, ratios, "o-", color="tab:green")
    ax2.set_xscale("log")
    ax2.set_xlabel("vocab size (log)")
    ax2.set_ylabel("compression ratio (bytes/token)")
    ax2.set_title("Compression ratio vs vocab size")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


SAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "def main(): user_id = 42",
    "SELECT id, name FROM users WHERE active = TRUE;",
    "https://example.com/api/v1/users/42",
    "replaying the song",
    "internationalization",
    "The cat sat on the mat.",
    "loss.backward()",
    "optimizer.zero_grad()",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Override the vocab sizes to train.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
    )
    args = parser.parse_args()

    if args.vocab_sizes is None:
        args.vocab_sizes = [288, 512, 1024, 2048, 4096] if args.full else [288, 512, 1024]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = DEFAULT_CORPUS

    print(f"Corpus size: {len(corpus.encode('utf-8'))} bytes")
    print(f"Vocab sizes to train: {args.vocab_sizes}")

    results: list[tuple[int, float, float]] = []
    visualizations: dict[int, list[tuple[str, str]]] = {}
    stats_table: list[dict] = []

    for vsize in args.vocab_sizes:
        print(f"\n=== Training BPE @ vocab_size={vsize} ===")
        tok = BPETokenizer()
        actual_size = tok.train(corpus, vocab_size=vsize, verbose=False)
        print(
            f"  requested vocab_size={vsize}, actual={actual_size} (learned {len(tok.merges)} merges)"
        )
        if actual_size < vsize:
            print(
                f"  (BPE stopped early: corpus exhausted multi-occurrence pairs at "
                f"vocab={actual_size}; further merges would only invent one-off tokens.)"
            )

        # Roundtrip sanity: encode -> decode should match original.
        for sample in SAMPLE_SENTENCES:
            assert tok.decode(tok.encode(sample)) == sample, (
                f"roundtrip failed at vocab_size={vsize} on {sample!r}"
            )

        # Average tokens per sample sentence + compression on corpus.
        token_counts = [len(tok.encode(s)) for s in SAMPLE_SENTENCES]
        avg_tokens = sum(token_counts) / len(token_counts)
        b, t, ratio = compression_ratio(corpus, tok)
        print(f"  avg tokens/sentence: {avg_tokens:.2f}")
        print(f"  corpus: {b} bytes -> {t} tokens; compression ratio = {ratio:.3f}")

        # Token frequency stats on the corpus.
        freq = token_frequency_stats(tok, corpus)
        print(
            f"  unique tokens used: {freq['unique_tokens_used']}  "
            f"median_freq: {freq['median_freq']}  "
            f"<5 freq frac: {freq['lt_5_frac']:.2%}  "
            f"==1 freq frac: {freq['eq_1_frac']:.2%}"
        )

        results.append((vsize, avg_tokens, ratio))
        stats_table.append(
            {
                "vocab_size": vsize,
                "avg_tokens": avg_tokens,
                "compression_ratio": ratio,
                **freq,
            }
        )

        # Visualization on 4 sample sentences.
        viz = [(s, visualize_tokens(tok, s)) for s in SAMPLE_SENTENCES[:4]]
        visualizations[vsize] = viz
        for sentence, rendered in viz:
            print(f"  {sentence!r}")
            print(f"    -> {rendered}")

    # === Outputs ===
    write_compression_plot(results, args.output_dir / "compression.png")

    log = args.output_dir / "run_log.txt"
    lines = [
        "# Project 3 run log",
        f"corpus_size_bytes: {len(corpus.encode('utf-8'))}",
        f"vocab_sizes      : {args.vocab_sizes}",
        "",
        "# per vocab size",
    ]
    for row in stats_table:
        lines.append(
            f"  vocab_size={row['vocab_size']}: "
            f"avg_tokens={row['avg_tokens']:.2f}  "
            f"compression={row['compression_ratio']:.3f}  "
            f"unique={row['unique_tokens_used']}  "
            f"<5freq={row['lt_5_frac']:.2%}  "
            f"==1freq={row['eq_1_frac']:.2%}"
        )
    lines.append("")
    lines.append("# token boundary visualizations (largest vocab size)")
    largest = max(visualizations.keys())
    for sentence, rendered in visualizations[largest]:
        lines.append(f"  {sentence}")
        lines.append(f"    {rendered}")
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nOutputs written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
