"""
Project 2: BREAK IT — destroy the embedding table's ability to distinguish characters.

We force every row of the embedding table to be the same vector. The forward
pass still runs (shapes are fine), but every character lookup returns the same
vector, so the model can no longer condition on which character it sees.

Expected: the broken model converges to a "global character frequency" predictor
and its loss plateaus far above the baseline.

Run:
    python break_it.py --tiny
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from build import (
    DEFAULT_NAMES,
    NeuralCharLM,
    build_neural_training_data,
    build_vocab,
    train_neural,
)


def collapse_embeddings(model: NeuralCharLM) -> None:
    """
    Force every embedding row to equal row 0 AND freeze C so the sabotage sticks.

    Without freezing, gradients flowing back through C re-introduce row
    distinctions within a few hundred steps. Freezing keeps the collapse
    intact so we can see the model's loss plateau without the embedding signal.
    """
    with torch.no_grad():
        model.C[:] = model.C[0]
    model.C.requires_grad_(False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
    )
    args = parser.parse_args()

    if args.steps is None:
        args.steps = 10_000 if args.full else 500

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    words = DEFAULT_NAMES
    _chars, stoi, _itos = build_vocab(words)
    V = len(stoi)

    X, Y = build_neural_training_data(words, stoi, args.block_size)
    n_train = int(X.shape[0] * 0.85)
    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]

    eval_every = max(1, args.steps // 10)

    # === Baseline ===
    torch.manual_seed(args.seed)
    baseline = NeuralCharLM(
        vocab_size=V,
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )
    base_train, base_val = train_neural(
        baseline,
        X_tr,
        Y_tr,
        X_val,
        Y_val,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=eval_every,
        seed=args.seed,
    )

    # === Broken: collapse all embedding rows to the same vector ===
    torch.manual_seed(args.seed)
    broken = NeuralCharLM(
        vocab_size=V,
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )
    collapse_embeddings(broken)
    broken_train, broken_val = train_neural(
        broken,
        X_tr,
        Y_tr,
        X_val,
        Y_val,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=eval_every,
        seed=args.seed,
    )

    print(f"{'mode':20s}  {'final train':>14s}  {'final val':>12s}")
    print(f"{'-' * 50}")
    print(f"{'baseline':20s}  {base_train[-1]:>14.4f}  {base_val[-1]:>12.4f}")
    print(f"{'collapsed embeds':20s}  {broken_train[-1]:>14.4f}  {broken_val[-1]:>12.4f}")

    # Check whether the embedding rows stayed identical during training.
    # If gradient flow doesn't preserve the collapse, broken model recovers.
    rows_max_diff = (broken.C - broken.C[0]).abs().max().item()
    print(f"\nmax row-to-row difference in broken model's C after training: {rows_max_diff:.6f}")
    if rows_max_diff > 0.01:
        print("(rows drifted — gradients re-introduced distinction; sabotage was overcome)")
    else:
        print("(rows stayed identical — model genuinely could not condition on character)")

    log = args.output_dir / "break_it_log.txt"
    log.write_text(
        "# Project 2 BREAK IT — collapse all embedding rows to the same vector\n"
        f"baseline:        train={base_train[-1]:.4f}  val={base_val[-1]:.4f}\n"
        f"collapsed C:     train={broken_train[-1]:.4f}  val={broken_val[-1]:.4f}\n"
        f"max row diff:    {rows_max_diff:.6f}\n"
        "\n"
        "Lesson: an embedding's job is to route distinct token IDs to distinct\n"
        "vectors. If every row of C is the same, the lookup C[X] returns identical\n"
        "vectors for every character — the network sees one input regardless of\n"
        "actual characters. The flattened context becomes a constant, the hidden\n"
        "layer can only encode 'how long until end-of-sequence', and the model\n"
        "regresses to a global character-frequency predictor.\n"
        "\n"
        "Note: if you let training continue, gradients flowing back through C\n"
        "will gradually re-introduce row distinctions. The 'collapse + retrain'\n"
        "demo above intentionally collapses BEFORE training so we can watch loss\n"
        "plateau higher than baseline.\n",
        encoding="utf-8",
    )
    print(f"\nLog written to {log.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
