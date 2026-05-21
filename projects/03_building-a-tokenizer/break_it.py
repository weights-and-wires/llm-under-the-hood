"""
Project 3: BREAK IT — vocab too small vs. vocab too large.

Two failure modes:

1. **vocab_size=256** (no merges, pure byte-level): sequences become 4x longer,
   the model wastes context reassembling common chunks.
2. **vocab_size=8192** (way too many for this tiny corpus): BPE stops early
   when pairs exhaust, but the merges it does learn become one-off "fossils"
   that won't get useful gradient signal in any downstream model.

Run:
    python break_it.py --tiny
"""

from __future__ import annotations

import argparse
from pathlib import Path

from build import (
    DEFAULT_CORPUS,
    SAMPLE_SENTENCES,
    BPETokenizer,
    compression_ratio,
    token_frequency_stats,
    visualize_tokens,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = DEFAULT_CORPUS

    # === Sane baseline: vocab=512 ===
    base = BPETokenizer()
    base.train(corpus, vocab_size=512)
    base_avg = sum(len(base.encode(s)) for s in SAMPLE_SENTENCES) / len(SAMPLE_SENTENCES)
    _, _, base_ratio = compression_ratio(corpus, base)
    base_freq = token_frequency_stats(base, corpus)

    # === Too small: vocab=256 (no merges; pure bytes) ===
    tiny = BPETokenizer()
    tiny.train(corpus, vocab_size=256)
    tiny_avg = sum(len(tiny.encode(s)) for s in SAMPLE_SENTENCES) / len(SAMPLE_SENTENCES)
    _, _, tiny_ratio = compression_ratio(corpus, tiny)

    # === Too large: vocab=8192 (will stop early but yields many rare tokens) ===
    huge = BPETokenizer()
    huge_actual = huge.train(corpus, vocab_size=8192)
    huge_avg = sum(len(huge.encode(s)) for s in SAMPLE_SENTENCES) / len(SAMPLE_SENTENCES)
    _, _, huge_ratio = compression_ratio(corpus, huge)
    huge_freq = token_frequency_stats(huge, corpus)

    print(f"{'mode':25s}  {'avg tokens/sent':>16s}  {'compression':>12s}  {'==1freq':>10s}")
    print("-" * 70)
    print(
        f"{'baseline (vocab=512)':25s}  {base_avg:>16.2f}  {base_ratio:>12.3f}  "
        f"{base_freq['eq_1_frac']:>10.1%}"
    )
    print(f"{'too small (vocab=256)':25s}  {tiny_avg:>16.2f}  {tiny_ratio:>12.3f}  {'-':>10s}")
    print(
        f"{'too large (req 8192)':25s}  {huge_avg:>16.2f}  {huge_ratio:>12.3f}  "
        f"{huge_freq['eq_1_frac']:>10.1%}"
    )
    print(f"\n(too-large requested 8192, got actual={huge_actual} after BPE exhausted pairs)")

    print("\nVisualization at vocab=256 (pure bytes — no merges):")
    print(f"  '{SAMPLE_SENTENCES[0]}'")
    print(f"  -> {visualize_tokens(tiny, SAMPLE_SENTENCES[0])}")

    print("\nVisualization at vocab=512 (sane baseline):")
    print(f"  '{SAMPLE_SENTENCES[0]}'")
    print(f"  -> {visualize_tokens(base, SAMPLE_SENTENCES[0])}")

    log = args.output_dir / "break_it_log.txt"
    log.write_text(
        "# Project 3 BREAK IT — vocab too small vs. too large\n\n"
        f"baseline (vocab=512):       avg_tokens={base_avg:.2f}  "
        f"compression={base_ratio:.3f}  ==1freq={base_freq['eq_1_frac']:.1%}\n"
        f"too small (vocab=256):       avg_tokens={tiny_avg:.2f}  "
        f"compression={tiny_ratio:.3f}\n"
        f"too large (req 8192, got {huge_actual}):  avg_tokens={huge_avg:.2f}  "
        f"compression={huge_ratio:.3f}  ==1freq={huge_freq['eq_1_frac']:.1%}\n"
        "\n"
        "Lessons:\n"
        "1. Too-small vocab (256, no merges): byte-level tokens force every common\n"
        "   word to be reassembled from scratch every time. Sentences are ~4x longer.\n"
        "   A model would waste context budget on local reassembly.\n"
        "\n"
        "2. Too-large vocab (8192 requested): BPE stops early because no pair\n"
        "   occurs more than once. Most of the late merges are one-off fragments\n"
        "   that wouldn't get reusable embeddings in any downstream model.\n"
        "\n"
        "The sweet spot exists because two costs pull in opposite directions:\n"
        "small vocab causes long sequences and wasted context; large vocab causes\n"
        "sparse token frequencies and weak embeddings. Modern tokenizers land in\n"
        "the tens of thousands — enough to compress common patterns, not so many\n"
        "that each piece becomes statistically lonely.\n",
        encoding="utf-8",
    )
    print(f"\nLog written to {log.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
