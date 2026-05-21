"""
Project 8: Flash Attention and Tiled Kernels — a CPU reference implementation.

We can't write a real CUDA kernel here, but we can demonstrate the key insight
of FlashAttention: you can compute the exact same attention output **without
ever materializing the full (T, T) score matrix**. The math is identical; only
the schedule is different.

Two implementations:

1. **Naive attention**: builds the (T, T) score matrix in memory, softmaxes,
   multiplies by V. Peak memory: O(T^2).
2. **Tiled attention**: processes Q in chunks of `q_block` rows. For each Q
   chunk, walks over K/V in chunks of `kv_block` columns, accumulating a
   running max + softmax-numerator. Peak intermediate memory: O(T * q_block).

This project verifies they produce identical outputs (up to FP error) and
prints the peak intermediate-tensor size for each.

Run:
    python build.py --tiny
    python build.py --full     # T=512 stress test
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def naive_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = True
) -> tuple[torch.Tensor, int]:
    """Standard attention: build (T,T) scores, softmax, mix V. Returns (output, peak_mem_floats)."""
    T, d_head = Q.shape
    scores = (Q @ K.T) / math.sqrt(d_head)  # (T, T)  ← THIS is what eats memory
    if causal:
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = weights @ V
    peak_mem = scores.numel() + weights.numel()  # (T,T) + (T,T) intermediate floats
    return out, peak_mem


def tiled_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    q_block: int = 32,
    kv_block: int = 32,
    causal: bool = True,
) -> tuple[torch.Tensor, int]:
    """
    Process Q in chunks of q_block; for each chunk walk over K/V in chunks of kv_block.

    Uses the online-softmax trick: maintain a running max `m` and running denominator `l`
    per query row, never building the full (T, T) score matrix.
    """
    T, d_head = Q.shape
    out = torch.zeros_like(Q)

    peak_intermediate = 0  # max floats held at once

    for i_start in range(0, T, q_block):
        i_end = min(i_start + q_block, T)
        Qi = Q[i_start:i_end]  # (qb, d_head)
        qb = Qi.shape[0]

        # Running max and softmax-denominator for this Q chunk's rows.
        m = torch.full((qb,), float("-inf"))
        l = torch.zeros(qb)
        Oi = torch.zeros(qb, d_head)

        for j_start in range(0, T, kv_block):
            j_end = min(j_start + kv_block, T)
            Kj = K[j_start:j_end]  # (kb, d_head)
            Vj = V[j_start:j_end]
            kb = Kj.shape[0]

            scores_ij = (Qi @ Kj.T) / math.sqrt(d_head)  # (qb, kb)
            if causal:
                # Apply causal mask: query position i can attend to key position j only if j <= i.
                row_idx = torch.arange(i_start, i_end).unsqueeze(1)
                col_idx = torch.arange(j_start, j_end).unsqueeze(0)
                mask = col_idx <= row_idx
                scores_ij = scores_ij.masked_fill(~mask, float("-inf"))

            # Online softmax update.
            mij = scores_ij.max(dim=-1).values  # (qb,)
            mi_new = torch.maximum(m, mij)
            # Rescale previous accumulator.
            alpha = torch.exp(m - mi_new)
            beta = torch.exp(scores_ij - mi_new.unsqueeze(-1))
            # Mask out positions that are -inf (otherwise NaN from exp(-inf - -inf)).
            beta = torch.nan_to_num(beta, nan=0.0)
            Oi = Oi * alpha.unsqueeze(-1) + beta @ Vj
            l = l * alpha + beta.sum(dim=-1)
            m = mi_new

            # Track peak intermediate
            peak_intermediate = max(peak_intermediate, scores_ij.numel() + Oi.numel())

        # Finalize: divide accumulator by denominator.
        out[i_start:i_end] = Oi / l.unsqueeze(-1)

    return out, peak_intermediate


def build_random_qkv(
    T: int, d_head: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(T, d_head, generator=g)
    K = torch.randn(T, d_head, generator=g)
    V = torch.randn(T, d_head, generator=g)
    return Q, K, V


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--T", type=int, default=None, help="Sequence length")
    parser.add_argument("--d-head", type=int, default=32)
    parser.add_argument("--q-block", type=int, default=32)
    parser.add_argument("--kv-block", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
    )
    args = parser.parse_args()

    if args.T is None:
        args.T = 512 if args.full else 128

    args.output_dir.mkdir(parents=True, exist_ok=True)

    Q, K, V = build_random_qkv(args.T, args.d_head, seed=args.seed)
    print(f"Sequence length T = {args.T}, d_head = {args.d_head}")
    print(f"Q,K,V each: ({args.T}, {args.d_head}) = {Q.numel()} floats each\n")

    # === Naive ===
    out_naive, peak_naive = naive_attention(Q, K, V, causal=True)
    print(f"naive  :  output_shape={tuple(out_naive.shape)}  peak_intermediate={peak_naive} floats")

    # === Tiled ===
    out_tiled, peak_tiled = tiled_attention(
        Q, K, V, q_block=args.q_block, kv_block=args.kv_block, causal=True
    )
    print(f"tiled  :  output_shape={tuple(out_tiled.shape)}  peak_intermediate={peak_tiled} floats")

    # === Correctness check ===
    max_diff = (out_naive - out_tiled).abs().max().item()
    mean_diff = (out_naive - out_tiled).abs().mean().item()
    print(f"\nmax |naive - tiled|  = {max_diff:.2e}")
    print(f"mean |naive - tiled| = {mean_diff:.2e}")
    print("(should be near zero — both compute the same attention)")

    # Memory ratio
    ratio = peak_naive / max(peak_tiled, 1)
    print(
        f"\nNaive peak intermediate: {peak_naive} floats  (= {peak_naive * 4 / 1024:.1f} KB at fp32)"
    )
    print(
        f"Tiled peak intermediate: {peak_tiled} floats  (= {peak_tiled * 4 / 1024:.1f} KB at fp32)"
    )
    print(f"Memory reduction: {ratio:.2f}x")

    log = args.output_dir / "run_log.txt"
    log.write_text(
        f"# Project 8 run log\n"
        f"T                       = {args.T}\n"
        f"d_head                  = {args.d_head}\n"
        f"q_block                 = {args.q_block}\n"
        f"kv_block                = {args.kv_block}\n"
        f"\n"
        f"naive peak intermediate = {peak_naive} floats ({peak_naive * 4 / 1024:.1f} KB fp32)\n"
        f"tiled peak intermediate = {peak_tiled} floats ({peak_tiled * 4 / 1024:.1f} KB fp32)\n"
        f"memory reduction        = {ratio:.2f}x\n"
        f"\n"
        f"max |naive - tiled|     = {max_diff:.2e}\n"
        f"mean |naive - tiled|    = {mean_diff:.2e}\n"
        f"\n"
        "Lesson: FlashAttention's actual contribution is *not* a new algorithm.\n"
        "It is a different scheduling of the same algorithm. By processing Q in\n"
        "chunks and tracking running softmax statistics (max + denominator),\n"
        "you never need to hold the full (T, T) score matrix in memory.\n"
        "\n"
        "On CPU this won't save wall time (numpy/torch already vectorize well),\n"
        "but on GPU the peak intermediate memory determines whether you can run\n"
        "longer contexts at all. T = 8192, d_head = 128:\n"
        "  naive intermediate = 8192*8192 = 64 MB per attention head per layer\n"
        "  tiled intermediate = 8192*64   = 0.5 MB per attention head per layer\n"
        "That difference is what makes long-context LLMs feasible.\n",
        encoding="utf-8",
    )
    print(f"\nOutputs written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
