"""
Project 5: BREAK IT — remove the residual connections.

In a pre-norm transformer block, the standard pattern is:
    x = x + attn(LayerNorm(x))
    x = x + mlp(LayerNorm(x))

We replace those with:
    x = attn(LayerNorm(x))
    x = mlp(LayerNorm(x))

The "+ x" is the residual connection. It lets the original signal pass through
unmodified — each sublayer learns a correction, not a replacement. Without it,
early-training noise from random sublayers wipes out the input signal entirely,
and the network struggles to learn anything coherent.

Run:
    python break_it.py --tiny
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from build import (
    DEFAULT_CORPUS,
    GPT,
    GPTConfig,
    char_tokenizer,
    encode,
    train_gpt,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
    )
    args = parser.parse_args()

    if args.steps is None:
        args.steps = 1000 if args.full else 200

    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = DEFAULT_CORPUS
    stoi, _itos, vocab_size = char_tokenizer(text)
    data = encode(text, stoi)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]

    cfg = GPTConfig(block_size=32, n_layers=2, n_heads=4, d_model=64)

    # === Baseline (with residuals) ===
    torch.manual_seed(args.seed)
    baseline = GPT(cfg, vocab_size=vocab_size, use_residual=True)
    base_train, base_val = train_gpt(
        baseline,
        train_data,
        val_data,
        steps=args.steps,
        batch_size=32,
        lr=3e-3,
        eval_every=max(1, args.steps // 10),
        seed=args.seed,
    )

    # === Broken (no residuals) ===
    torch.manual_seed(args.seed)
    broken = GPT(cfg, vocab_size=vocab_size, use_residual=False)
    broken_train, broken_val = train_gpt(
        broken,
        train_data,
        val_data,
        steps=args.steps,
        batch_size=32,
        lr=3e-3,
        eval_every=max(1, args.steps // 10),
        seed=args.seed,
    )

    uniform = math.log(vocab_size)
    print(f"\nUniform baseline (random guessing): {uniform:.4f}")
    print(f"\n{'mode':25s}  {'final train':>14s}  {'final val':>12s}")
    print("-" * 55)
    print(f"{'baseline (residual)':25s}  {base_train[-1]:>14.4f}  {base_val[-1]:>12.4f}")
    print(f"{'broken (no residual)':25s}  {broken_train[-1]:>14.4f}  {broken_val[-1]:>12.4f}")

    log = args.output_dir / "break_it_log.txt"
    log.write_text(
        f"# Project 5 BREAK IT — remove residual connections\n\n"
        f"uniform baseline      : {uniform:.4f}\n"
        f"baseline (residual)   : train={base_train[-1]:.4f}  val={base_val[-1]:.4f}\n"
        f"broken (no residual)  : train={broken_train[-1]:.4f}  val={broken_val[-1]:.4f}\n"
        "\n"
        "Lesson: each residual connection lets the original signal flow through\n"
        "unmodified. The sublayer learns a CORRECTION, not a REPLACEMENT. When you\n"
        "remove residuals:\n"
        "  - Random init of attention/MLP wipes out the input signal completely\n"
        "    before training has done anything useful.\n"
        "  - Deep stacks become untrainable — each layer destroys what the\n"
        "    previous layer learned to express.\n"
        "  - LayerNorm pre-attention can't compensate; it can normalize but it\n"
        "    can't restore the lost signal.\n"
        "\n"
        "This is the single biggest architectural reason transformers go deep\n"
        "in the first place. Without residual connections, deeper = worse.\n",
        encoding="utf-8",
    )
    print(f"\nLog written to {log.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
