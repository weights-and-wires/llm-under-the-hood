"""
Project 4: Attention from Scratch — complete working build.

Builds single-head causal self-attention, then multi-head attention, both with
explicit Q/K/V projections. Demonstrates the 1/sqrt(d_head) scaling and prints
per-head attention entropy as a diagnostic.

Run:
    python build.py --tiny
    python build.py --full
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

# A short pretend "sentence" — we don't need a real tokenizer for this project.
TOY_TOKENS = ["the", "cat", "sat", "on", "the", "mat", ".", "and"]


def make_input_embeddings(T: int, d_model: int, seed: int = 0) -> torch.Tensor:
    """Random per-position embeddings — stand-in for token embeddings."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, d_model, generator=g)


def causal_mask(T: int) -> torch.Tensor:
    """Lower-triangular mask: position i can attend to positions <= i."""
    return torch.tril(torch.ones(T, T))


def single_head_attention(
    x: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    mask: torch.Tensor,
    scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (output, weights) where weights are the (T, T) attention matrix."""
    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V
    scores = Q @ K.T
    if scale:
        d_head = Q.shape[-1]
        scores = scores / math.sqrt(d_head)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = weights @ V
    return out, weights


def multi_head_attention(
    x: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    W_O: torch.Tensor,
    n_heads: int,
    mask: torch.Tensor,
    scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (output, per_head_weights of shape (H, T, T))."""
    T, d_model = x.shape
    d_head = d_model // n_heads

    Q = x @ W_Q  # (T, d_model)
    K = x @ W_K
    V = x @ W_V

    # Reshape: (T, d_model) -> (T, H, d_head) -> (H, T, d_head)
    Q = Q.view(T, n_heads, d_head).transpose(0, 1)
    K = K.view(T, n_heads, d_head).transpose(0, 1)
    V = V.view(T, n_heads, d_head).transpose(0, 1)

    scores = Q @ K.transpose(-2, -1)  # (H, T, T)
    if scale:
        scores = scores / math.sqrt(d_head)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    head_out = weights @ V  # (H, T, d_head)

    out = head_out.transpose(0, 1).contiguous().view(T, d_model)
    out = out @ W_O
    return out, weights


def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """For each row of `weights` (a probability vector), compute -sum(p*log(p))."""
    # Add tiny eps to avoid log(0); zeroed positions in causal mask contribute 0 anyway.
    log_w = torch.log(weights.clamp_min(1e-12))
    return -(weights * log_w).sum(dim=-1)


def init_projections(
    d_model: int, n_heads: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    scale = 1.0 / math.sqrt(d_model)
    W_Q = torch.randn(d_model, d_model, generator=g) * scale
    W_K = torch.randn(d_model, d_model, generator=g) * scale
    W_V = torch.randn(d_model, d_model, generator=g) * scale
    W_O = torch.randn(d_model, d_model, generator=g) * scale
    return W_Q, W_K, W_V, W_O


def write_heatmap(weights: torch.Tensor, tokens: list[str], path: Path, title: str) -> None:
    """Plot per-head attention heatmaps (H, T, T) into a grid."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if weights.ndim == 2:
        weights = weights.unsqueeze(0)
    H = weights.shape[0]
    cols = min(H, 4)
    rows = (H + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for h in range(H):
        ax = axes[h // cols][h % cols]
        im = ax.imshow(weights[h].numpy(), cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"head {h}")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, fontsize=8)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for h in range(H, rows * cols):
        axes[h // cols][h % cols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
    )
    args = parser.parse_args()

    if args.full:
        args.d_model = 64
        args.n_heads = 8

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens = TOY_TOKENS
    T = len(tokens)
    d_model = args.d_model
    d_head = d_model // args.n_heads

    print(f"Tokens: {tokens}")
    print(f"T={T}  d_model={d_model}  n_heads={args.n_heads}  d_head={d_head}")

    x = make_input_embeddings(T, d_model, seed=args.seed)
    mask = causal_mask(T)
    W_Q, W_K, W_V, W_O = init_projections(d_model, args.n_heads, seed=args.seed)

    # === Single head — show the shapes ===
    print("\n=== Single-head attention ===")
    out1, weights1 = single_head_attention(x, W_Q, W_K, W_V, mask, scale=True)
    print(f"  Q,K,V shape: ({T}, {d_model})")
    print(f"  scores (post-mask, post-softmax) shape: {tuple(weights1.shape)}")
    print(f"  output shape: {tuple(out1.shape)}")
    print(f"  attention matrix (one head):\n{weights1.numpy().round(3)}")

    # === Multi-head ===
    print("\n=== Multi-head attention ===")
    out_mh, weights_mh = multi_head_attention(x, W_Q, W_K, W_V, W_O, args.n_heads, mask, scale=True)
    print(f"  multi-head output shape: {tuple(out_mh.shape)}")
    print(f"  weights shape (H, T, T): {tuple(weights_mh.shape)}")

    # Per-head entropy
    entropy = attention_entropy(weights_mh)  # (H, T)
    mean_entropy_per_head = entropy.mean(dim=-1)
    max_entropy_possible = math.log(T)  # uniform over T positions
    print("\nPer-head mean entropy (lower = more concentrated):")
    for h, e in enumerate(mean_entropy_per_head.tolist()):
        bar = "#" * int(20 * e / max_entropy_possible)
        print(f"  head {h}: {e:.3f}  {bar:<20s}  (max possible: {max_entropy_possible:.3f})")

    # === Outputs ===
    write_heatmap(
        weights_mh,
        tokens,
        args.output_dir / "attention_heatmaps.png",
        title=f"Multi-head attention (T={T}, H={args.n_heads}, scaled)",
    )

    log = args.output_dir / "run_log.txt"
    lines = [
        "# Project 4 run log",
        f"tokens     : {tokens}",
        f"T          : {T}",
        f"d_model    : {d_model}",
        f"n_heads    : {args.n_heads}",
        f"d_head     : {d_head}",
        "",
        "# single-head attention weights (causal mask + scaled dot product)",
        f"{weights1.numpy().round(3)}",
        "",
        f"# per-head mean entropy (max possible = log(T) = {max_entropy_possible:.3f})",
    ]
    for h, e in enumerate(mean_entropy_per_head.tolist()):
        lines.append(f"  head {h}: {e:.4f}")
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nOutputs written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
