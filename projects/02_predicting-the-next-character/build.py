"""
Project 2: Predicting the Next Character — complete working build.

Two character-level language models on a tiny names corpus:

1. A bigram counting model (no learning — just a count table normalized to probabilities).
2. A neural MLP with learned character embeddings and a fixed context window.

Both produce samples and a negative log-likelihood (NLL) score on held-out data.

Run:
    python build.py --tiny             # 500 training steps, <30s on CPU
    python build.py --full             # 10000 steps, ~3 min on CPU
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

# A small built-in names corpus (no network, no file fetch).
# 80 short names; enough for the bigram to make sense and the neural model
# to clearly outperform it.
DEFAULT_NAMES = [
    "emma",
    "olivia",
    "ava",
    "isabella",
    "sophia",
    "charlotte",
    "mia",
    "amelia",
    "harper",
    "evelyn",
    "abigail",
    "emily",
    "elizabeth",
    "mila",
    "ella",
    "avery",
    "sofia",
    "camila",
    "aria",
    "scarlett",
    "victoria",
    "madison",
    "luna",
    "grace",
    "chloe",
    "penelope",
    "layla",
    "riley",
    "zoey",
    "nora",
    "lily",
    "eleanor",
    "hannah",
    "lillian",
    "addison",
    "aubrey",
    "ellie",
    "stella",
    "natalie",
    "zoe",
    "leah",
    "hazel",
    "violet",
    "aurora",
    "savannah",
    "audrey",
    "brooklyn",
    "bella",
    "claire",
    "skylar",
    "lucy",
    "paisley",
    "everly",
    "anna",
    "caroline",
    "nova",
    "genesis",
    "emilia",
    "kennedy",
    "samantha",
    "maya",
    "willow",
    "kinsley",
    "naomi",
    "aaliyah",
    "elena",
    "sarah",
    "ariana",
    "allison",
    "gabriella",
    "alice",
    "madelyn",
    "cora",
    "ruby",
    "eva",
    "serenity",
    "autumn",
    "adeline",
    "hailey",
    "gianna",
]


def build_vocab(words: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Vocabulary is all unique characters in the corpus, plus a special `.` token."""
    chars = sorted(set("".join(words)))
    chars = ["."] + chars  # `.` as start/end token
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


def build_bigram_counts(words: list[str], stoi: dict[str, int]) -> torch.Tensor:
    """Count transitions. Rows = current char, cols = next char. Returns int32 (V, V)."""
    V = len(stoi)
    N = torch.zeros((V, V), dtype=torch.int32)
    for word in words:
        chs = ["."] + list(word) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            i, j = stoi[ch1], stoi[ch2]
            N[i, j] += 1
    return N


def bigram_probs(N: torch.Tensor, smoothing: int = 1) -> torch.Tensor:
    """Convert counts to probabilities; add `+smoothing` to every cell to avoid zeros."""
    P = (N + smoothing).float()
    P = P / P.sum(dim=1, keepdim=True)
    return P


def bigram_nll(P: torch.Tensor, words: list[str], stoi: dict[str, int]) -> float:
    """Negative log-likelihood per character on `words`. Lower is better."""
    log_likelihood = 0.0
    n = 0
    for word in words:
        chs = ["."] + list(word) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            i, j = stoi[ch1], stoi[ch2]
            log_likelihood += math.log(P[i, j].item())
            n += 1
    return -log_likelihood / n


def bigram_sample(
    P: torch.Tensor, itos: dict[int, str], n_samples: int, seed: int = 0
) -> list[str]:
    g = torch.Generator().manual_seed(seed)
    samples: list[str] = []
    start = 0  # `.` is index 0
    for _ in range(n_samples):
        ix = start
        out: list[str] = []
        while True:
            ix = int(torch.multinomial(P[ix], 1, replacement=True, generator=g).item())
            ch = itos[ix]
            if ch == ".":
                break
            out.append(ch)
            if len(out) > 30:  # safety bound
                break
        samples.append("".join(out))
    return samples


def build_neural_training_data(
    words: list[str], stoi: dict[str, int], block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each word, create (context, target) pairs with block_size context."""
    X: list[list[int]] = []
    Y: list[int] = []
    for word in words:
        context = [stoi["."]] * block_size
        for ch in list(word) + ["."]:
            ix = stoi[ch]
            X.append(context.copy())
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


class NeuralCharLM:
    """A minimal MLP language model with learned character embeddings.

    Parameters: C (embedding table), W1/b1 (hidden), W2/b2 (output).
    No PyTorch nn.Module — we keep tensors explicit so the reader sees every weight.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embed_dim: int,
        hidden_size: int,
        seed: int = 0,
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        self.V = vocab_size
        self.block_size = block_size
        self.C = torch.randn((vocab_size, embed_dim), generator=g).requires_grad_(True)
        self.W1 = (
            torch.randn((block_size * embed_dim, hidden_size), generator=g)
            * (1.0 / math.sqrt(block_size * embed_dim))
        ).requires_grad_(True)
        self.b1 = torch.zeros(hidden_size, requires_grad=True)
        self.W2 = (
            torch.randn((hidden_size, vocab_size), generator=g) * (1.0 / math.sqrt(hidden_size))
        ).requires_grad_(True)
        self.b2 = torch.zeros(vocab_size, requires_grad=True)

    def parameters(self) -> list[torch.Tensor]:
        return [self.C, self.W1, self.b1, self.W2, self.b2]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        emb = self.C[X]  # (B, block_size, embed_dim)
        x = emb.view(emb.shape[0], -1)
        h = torch.tanh(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits

    def loss(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.forward(X), Y)

    @torch.no_grad()
    def sample(
        self,
        itos: dict[int, str],
        stoi: dict[str, int],
        n_samples: int,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> list[str]:
        g = torch.Generator().manual_seed(seed)
        samples: list[str] = []
        for _ in range(n_samples):
            context = [stoi["."]] * self.block_size
            out: list[str] = []
            while True:
                x = torch.tensor([context], dtype=torch.long)
                logits = self.forward(x)
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=1)
                ix = int(torch.multinomial(probs, 1, generator=g).item())
                ch = itos[ix]
                if ch == ".":
                    break
                out.append(ch)
                context = context[1:] + [ix]
                if len(out) > 30:
                    break
            samples.append("".join(out))
        return samples


def train_neural(
    model: NeuralCharLM,
    X_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    g = torch.Generator().manual_seed(seed)
    train_history: list[float] = []
    val_history: list[float] = []
    n = X_tr.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=g)
        Xb, Yb = X_tr[idx], Y_tr[idx]
        loss = model.loss(Xb, Yb)
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        loss.backward()
        for p in model.parameters():
            if p.grad is None:
                continue  # parameter was frozen (e.g. via requires_grad=False)
            with torch.no_grad():
                p -= lr * p.grad  # type: ignore[operator]
        if step % eval_every == 0 or step == steps - 1:
            train_history.append(float(loss.item()))
            with torch.no_grad():
                val_history.append(float(model.loss(X_val, Y_val).item()))
    return train_history, val_history


def write_loss_curve_png(
    train_hist: list[float], val_hist: list[float], steps_per_point: int, path: Path
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    xs = [i * steps_per_point for i in range(len(train_hist))]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, train_hist, label="train")
    ax.plot(xs, val_hist, label="val")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Project 2: neural char-LM training")
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
    chars, stoi, itos = build_vocab(words)
    V = len(chars)
    print(f"Corpus: {len(words)} names, vocab size {V}")
    print(f"Vocab: {chars}")

    # === Bigram counting model ===
    N = build_bigram_counts(words, stoi)
    P = bigram_probs(N, smoothing=1)
    bigram_loss = bigram_nll(P, words, stoi)
    print(f"\nBigram NLL (full corpus): {bigram_loss:.4f}")
    bigram_samples = bigram_sample(P, itos, n_samples=10, seed=args.seed)
    print("Bigram samples:", bigram_samples)

    # === Neural model with embeddings ===
    X, Y = build_neural_training_data(words, stoi, args.block_size)
    n_total = X.shape[0]
    n_train = int(n_total * 0.85)
    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]
    print(f"\nNeural training pairs: {n_train} train / {n_total - n_train} val")

    model = NeuralCharLM(
        vocab_size=V,
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Neural model parameters: {n_params}")

    eval_every = max(1, args.steps // 20)
    train_hist, val_hist = train_neural(
        model,
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
    print(
        f"\nNeural after {args.steps} steps: "
        f"train_loss={train_hist[-1]:.4f}  val_loss={val_hist[-1]:.4f}"
    )

    # Temperature sweep — generate samples at several temperatures.
    print("\nNeural samples at varied temperatures:")
    temp_samples: dict[float, list[str]] = {}
    for temp in [0.5, 1.0, 1.5]:
        s = model.sample(itos, stoi, n_samples=8, temperature=temp, seed=args.seed)
        temp_samples[temp] = s
        print(f"  T={temp}: {s}")

    # === Outputs ===
    write_loss_curve_png(
        train_hist, val_hist, steps_per_point=eval_every, path=args.output_dir / "loss_curve.png"
    )

    log = args.output_dir / "run_log.txt"
    lines = [
        "# Project 2 run log",
        f"corpus_size      : {len(words)} names",
        f"vocab_size       : {V}",
        f"block_size       : {args.block_size}",
        f"embed_dim        : {args.embed_dim}",
        f"hidden_size      : {args.hidden_size}",
        f"steps            : {args.steps}",
        "",
        "# bigram model (no learning, just counts + smoothing)",
        f"bigram_NLL       : {bigram_loss:.4f}",
        f"bigram_samples   : {bigram_samples}",
        "",
        "# neural model",
        f"n_parameters     : {n_params}",
        f"train_loss_final : {train_hist[-1]:.4f}",
        f"val_loss_final   : {val_hist[-1]:.4f}",
        "",
        "# temperature sweep",
    ]
    for temp, samples in temp_samples.items():
        lines.append(f"T={temp}:   {samples}")
    lines += [
        "",
        "# comparison",
        f"  bigram NLL:  {bigram_loss:.4f}",
        f"  neural val:  {val_hist[-1]:.4f}",
        f"  delta:       {bigram_loss - val_hist[-1]:+.4f}",
    ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nOutputs written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
