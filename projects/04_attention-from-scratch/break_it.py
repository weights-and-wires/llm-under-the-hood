"""
Project 4: BREAK IT — remove the 1/sqrt(d_head) scaling.

When raw scores get large in magnitude, softmax saturates: one entry per row
gets almost all the probability, the rest get almost none. Attention entropy
collapses, and gradients through the softmax become tiny. Training would
look fine for the first few steps then stop learning.

Run:
    python break_it.py --tiny
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from build import (
    TOY_TOKENS,
    attention_entropy,
    causal_mask,
    init_projections,
    make_input_embeddings,
    multi_head_attention,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens = TOY_TOKENS
    T = len(tokens)
    d_model = args.d_model
    d_head = d_model // args.n_heads

    x = make_input_embeddings(T, d_model, seed=args.seed)
    mask = causal_mask(T)
    W_Q, W_K, W_V, W_O = init_projections(d_model, args.n_heads, seed=args.seed)

    # === Scaled (baseline) ===
    _, w_scaled = multi_head_attention(x, W_Q, W_K, W_V, W_O, args.n_heads, mask, scale=True)
    ent_scaled = attention_entropy(w_scaled).mean(dim=-1)

    # === Unscaled (broken) ===
    _, w_unscaled = multi_head_attention(x, W_Q, W_K, W_V, W_O, args.n_heads, mask, scale=False)
    ent_unscaled = attention_entropy(w_unscaled).mean(dim=-1)

    # === Even worse: bigger d_head + unscaled ===
    big_d = 256
    big_x = make_input_embeddings(T, big_d, seed=args.seed)
    big_W_Q, big_W_K, big_W_V, big_W_O = init_projections(big_d, args.n_heads, seed=args.seed)
    _, w_big_unscaled = multi_head_attention(
        big_x, big_W_Q, big_W_K, big_W_V, big_W_O, args.n_heads, mask, scale=False
    )
    ent_big_unscaled = attention_entropy(w_big_unscaled).mean(dim=-1)

    max_ent = math.log(T)
    print(f"Max possible entropy per row (uniform over {T} positions): {max_ent:.3f}")
    print(f"d_head = {d_head} for first two rows, d_head = {big_d // args.n_heads} for third\n")
    print(f"{'mode':40s}  {'min':>6s}  {'mean':>6s}  {'max':>6s}")
    print("-" * 70)

    def row(label: str, e: torch.Tensor) -> None:
        print(
            f"{label:40s}  {e.min().item():>6.3f}  {e.mean().item():>6.3f}  {e.max().item():>6.3f}"
        )

    row(f"scaled (d_head={d_head})", ent_scaled)
    row(f"unscaled (d_head={d_head})", ent_unscaled)
    row(f"unscaled, d_head={big_d // args.n_heads} (catastrophic)", ent_big_unscaled)

    print("\nHead 0 attention row for position 7 of the unscaled, large-d_head model:")
    print(f"  {w_big_unscaled[0, 7].numpy().round(4)}")
    print("(Note: probability is nearly all on one position — softmax saturated.)")

    log = args.output_dir / "break_it_log.txt"
    log.write_text(
        f"# Project 4 BREAK IT — remove 1/sqrt(d_head) scaling\n\n"
        f"max possible entropy: {max_ent:.3f} (uniform over T={T})\n\n"
        f"scaled, d_head={d_head}              mean_entropy={ent_scaled.mean().item():.4f}\n"
        f"unscaled, d_head={d_head}            mean_entropy={ent_unscaled.mean().item():.4f}\n"
        f"unscaled, d_head={big_d // args.n_heads}            "
        f"mean_entropy={ent_big_unscaled.mean().item():.4f}\n"
        "\n"
        "Lesson: dot products of d-dimensional vectors grow with d. Unscaled scores\n"
        "saturate softmax, collapsing each row to one-hot. Gradients through that\n"
        "saturated softmax are vanishingly small for the suppressed positions, so\n"
        "the model can't learn to attend to the right place. The 1/sqrt(d_head)\n"
        "factor keeps scores in a useful magnitude range. It is NOT cosmetic.\n",
        encoding="utf-8",
    )
    print(f"\nLog written to {log.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
