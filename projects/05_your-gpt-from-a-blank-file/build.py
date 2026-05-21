"""
Project 5: Your GPT from a Blank File — complete working build.

A tiny GPT-style language model assembled from scratch with PyTorch nn.Module.
Token embedding + position embedding + N transformer blocks (causal attention +
MLP, pre-norm, residual connections) + final layer norm + tied LM head.

Train on a small built-in corpus, then sample.

Run:
    python build.py --tiny      # 200 steps, ~30s on CPU
    python build.py --full      # 5000 steps, ~10 min on CPU
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# A small built-in corpus to train on — fragments meant to be Shakespeare-shaped.
DEFAULT_CORPUS = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved, resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them.

Let us revenge this with our pikes, ere we become rakes:
for the gods know I speak this in hunger for bread,
not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report fort, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
is, even till the altitude of his virtue.
"""


@dataclass
class GPTConfig:
    block_size: int = 32
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 64
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-norm transformer block: attention then MLP, each with residual + LayerNorm."""

    def __init__(self, cfg: GPTConfig, use_residual: bool = True) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        else:
            # BREAK IT mode: replace residuals with the raw sublayer output.
            x = self.attn(self.ln1(x))
            x = self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig, vocab_size: int, use_residual: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg, use_residual=use_residual) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        # Weight tying: share embedding with output head.
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

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def char_tokenizer(text: str) -> tuple[dict[str, int], dict[int, str], int]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos, len(chars)


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def decode(ids: list[int] | torch.Tensor, itos: dict[int, str]) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return "".join(itos[i] for i in ids)


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def train_gpt(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    g = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_hist: list[float] = []
    val_hist: list[float] = []
    for step in range(steps):
        x, y = get_batch(train_data, model.cfg.block_size, batch_size, g)
        _, loss = model(x, y)
        optimizer.zero_grad()
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % eval_every == 0 or step == steps - 1:
            train_hist.append(float(loss.item()))
            with torch.no_grad():
                vx, vy = get_batch(val_data, model.cfg.block_size, batch_size, g)
                _, vloss = model(vx, vy)
                val_hist.append(float(vloss.item()))
    return train_hist, val_hist


def write_loss_curve(
    train_hist: list[float], val_hist: list[float], eval_every: int, path: Path
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    xs = [i * eval_every for i in range(len(train_hist))]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, train_hist, label="train")
    ax.plot(xs, val_hist, label="val")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Project 5: tiny GPT training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
    )
    args = parser.parse_args()

    if args.steps is None:
        args.steps = 5000 if args.full else 200

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    text = DEFAULT_CORPUS
    stoi, itos, vocab_size = char_tokenizer(text)
    data = encode(text, stoi)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]
    print(f"Corpus: {len(text)} chars, vocab_size={vocab_size}")
    print(f"Train tokens: {len(train_data)}  Val tokens: {len(val_data)}")

    cfg = GPTConfig(
        block_size=args.block_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
    )
    model = GPT(cfg, vocab_size=vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    # Account for weight tying — embedding and lm_head share weights.
    n_params -= model.token_embedding.weight.numel()
    print(f"Model parameters (after weight tying): ~{n_params}")

    eval_every = max(1, args.steps // 10)
    print(f"\nTraining for {args.steps} steps ...")
    train_hist, val_hist = train_gpt(
        model,
        train_data,
        val_data,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=eval_every,
        seed=args.seed,
    )
    print(
        f"Final: train_loss={train_hist[-1]:.4f}  val_loss={val_hist[-1]:.4f}  "
        f"(uniform would be log(vocab)={math.log(vocab_size):.4f})"
    )

    # === Generate samples ===
    model.eval()
    print("\n=== Sample generations (temperature=1.0) ===")
    prompt = encode("First Citizen:\n", stoi).unsqueeze(0)
    samples = []
    for _ in range(3):
        out_ids = model.generate(prompt, max_new_tokens=120, temperature=1.0)
        out_text = decode(out_ids[0], itos)
        samples.append(out_text)
        print(out_text)
        print("-" * 40)

    write_loss_curve(train_hist, val_hist, eval_every, args.output_dir / "loss_curve.png")
    log = args.output_dir / "run_log.txt"
    lines = [
        "# Project 5 run log",
        f"corpus_chars     : {len(text)}",
        f"vocab_size       : {vocab_size}",
        f"block_size       : {args.block_size}",
        f"d_model          : {args.d_model}",
        f"n_heads          : {args.n_heads}",
        f"n_layers         : {args.n_layers}",
        f"steps            : {args.steps}",
        f"n_parameters     : {n_params}",
        f"train_loss_final : {train_hist[-1]:.4f}",
        f"val_loss_final   : {val_hist[-1]:.4f}",
        f"uniform_baseline : {math.log(vocab_size):.4f}",
        "",
        "# sample generations (3 samples, temperature=1.0)",
    ]
    for i, sample in enumerate(samples):
        lines.append(f"--- sample {i + 1} ---")
        lines.append(sample)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nOutputs written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
