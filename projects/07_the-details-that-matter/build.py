"""
Project 7: The Details That Matter — RMSNorm and SwiGLU as alternatives.

Two architectural details that show up in modern LLMs (Llama, Mistral, etc.)
as replacements for the LayerNorm + GELU from the earlier transformer recipe:

1. **RMSNorm** — drops LayerNorm's mean-centering. Just normalize by root-mean-square.
   Same regularization effect at ~half the FLOPs.

2. **SwiGLU** — replaces the MLP's `Linear -> GELU -> Linear` with a gated form:
   `(silu(x @ W1) * (x @ W2)) @ W3`. The gating term learns which features to
   pass through. Slightly more parameters; consistently better in modern LLMs.

This project trains a tiny GPT in four configurations and measures the
difference: {LayerNorm, RMSNorm} x {GELU MLP, SwiGLU MLP}.

Run:
    python build.py --tiny      # ~30s on CPU
    python build.py --full
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_5 = Path(__file__).resolve().parent.parent / "05_your-gpt-from-a-blank-file"


def _load_project_5():
    spec = importlib.util.spec_from_file_location("project_05_build", PROJECT_5 / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["project_05_build"] = module
    spec.loader.exec_module(module)
    return module


p5 = _load_project_5()


class RMSNorm(nn.Module):
    """Root-mean-square layer norm. No mean centering; only divide by RMS."""

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class SwiGLU(nn.Module):
    """Gated MLP: out = (silu(x @ W1) * (x @ W2)) @ W3.

    Standard GELU MLP: d -> 4d -> d   (= 8 d^2 params)
    SwiGLU:           d -> 4d (gate) + d -> 4d (proj), then 4d -> d
                       = 4d^2 + 4d^2 + 4d^2 = 12d^2 params (with full 4d hidden)

    To keep parameter count roughly equal to a GELU MLP, we use 8/3 * d for the
    hidden dim, which gives roughly 8 d^2 params total. We use floor to keep
    things divisible.
    """

    def __init__(self, d: int, hidden_mult: float = 8.0 / 3.0) -> None:
        super().__init__()
        d_hidden = int(d * hidden_mult)
        self.gate_proj = nn.Linear(d, d_hidden, bias=False)
        self.up_proj = nn.Linear(d, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ModernBlock(nn.Module):
    """Transformer block with configurable norm and MLP."""

    def __init__(self, cfg: p5.GPTConfig, norm_type: str = "ln", mlp_type: str = "gelu") -> None:
        super().__init__()
        Norm = RMSNorm if norm_type == "rms" else nn.LayerNorm
        self.ln1 = Norm(cfg.d_model)
        self.attn = p5.CausalSelfAttention(cfg)
        self.ln2 = Norm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model) if mlp_type == "swiglu" else p5.MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ModernGPT(nn.Module):
    """Tiny GPT with configurable normalization and MLP variants."""

    def __init__(
        self,
        cfg: p5.GPTConfig,
        vocab_size: int,
        norm_type: str = "ln",
        mlp_type: str = "gelu",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        Norm = RMSNorm if norm_type == "rms" else nn.LayerNorm
        self.token_embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [ModernBlock(cfg, norm_type=norm_type, mlp_type=mlp_type) for _ in range(cfg.n_layers)]
        )
        self.ln_f = Norm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss


def train_variant(
    norm_type: str,
    mlp_type: str,
    vocab_size: int,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    steps: int,
    seed: int = 0,
) -> tuple[ModernGPT, float, float]:
    torch.manual_seed(seed)
    cfg = p5.GPTConfig(block_size=32, n_layers=2, n_heads=4, d_model=64)
    model = ModernGPT(cfg, vocab_size=vocab_size, norm_type=norm_type, mlp_type=mlp_type)
    g = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(steps):
        x, y = p5.get_batch(train_data, cfg.block_size, 32, g)
        _, loss = model(x, y)
        optimizer.zero_grad()
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    # Final eval
    with torch.no_grad():
        _, train_loss = model(*p5.get_batch(train_data, cfg.block_size, 32, g))
        _, val_loss = model(*p5.get_batch(val_data, cfg.block_size, 32, g))
    return model, float(train_loss.item()), float(val_loss.item())  # type: ignore[union-attr]


def count_params(model: ModernGPT) -> int:
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


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
        args.steps = 2000 if args.full else 200

    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = p5.DEFAULT_CORPUS
    stoi, _, vocab_size = p5.char_tokenizer(text)
    data = p5.encode(text, stoi)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]

    configs = [
        ("ln", "gelu"),
        ("rms", "gelu"),
        ("ln", "swiglu"),
        ("rms", "swiglu"),
    ]
    results = []
    for norm_type, mlp_type in configs:
        model, train_loss, val_loss = train_variant(
            norm_type,
            mlp_type,
            vocab_size,
            train_data,
            val_data,
            steps=args.steps,
            seed=args.seed,
        )
        n_params = count_params(model)
        results.append((norm_type, mlp_type, n_params, train_loss, val_loss))
        print(
            f"  {norm_type:5s} + {mlp_type:8s}  params={n_params:>6d}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}"
        )

    print(f"\nUniform baseline: {math.log(vocab_size):.4f}")
    print(f"\n{'norm':6s}  {'mlp':10s}  {'params':>8s}  {'train':>8s}  {'val':>8s}")
    print("-" * 50)
    for norm_type, mlp_type, n_params, train_loss, val_loss in results:
        print(
            f"{norm_type:6s}  {mlp_type:10s}  {n_params:>8d}  {train_loss:>8.4f}  {val_loss:>8.4f}"
        )

    # === Outputs ===
    log = args.output_dir / "run_log.txt"
    lines = [
        "# Project 7 run log",
        f"steps: {args.steps}",
        f"uniform baseline: {math.log(vocab_size):.4f}",
        "",
        f"{'norm':6s}  {'mlp':10s}  {'params':>8s}  {'train':>8s}  {'val':>8s}",
    ]
    for norm_type, mlp_type, n_params, train_loss, val_loss in results:
        lines.append(
            f"{norm_type:6s}  {mlp_type:10s}  {n_params:>8d}  {train_loss:>8.4f}  {val_loss:>8.4f}"
        )
    lines.append("")
    lines.append("Lessons:")
    lines.append("- RMSNorm: same regularization effect as LayerNorm, ~half the FLOPs.")
    lines.append("- SwiGLU: gating learns which features to pass through; modest extra")
    lines.append("  params (we scaled hidden_mult=8/3 to roughly match GELU MLP size).")
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nOutputs written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
