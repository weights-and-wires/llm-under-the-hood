"""
Project 6: From Prototype to nanoGPT — production-shape refinements.

Starts from the Project 5 prototype and applies two nanoGPT-style improvements:

1. **Parameter groups for weight decay**: weights get weight decay, biases and
   LayerNorm parameters do not. They have different roles, so the optimizer
   should treat them differently.

2. **Scaled residual initialization**: projections that sit on residual branches
   (attention output proj and MLP output proj) get initialized with reduced
   variance. In a deep residual stack each block adds to the running hidden
   state; if every addition is full-scale, the hidden state drifts upward
   layer by layer.

Trains side-by-side against a baseline (default init + uniform decay) and
reports the difference.

Run:
    python build.py --tiny
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

# Import Project 5's building blocks. We reuse its GPT class wholesale, then
# override init and optimizer.
PROJECT_5 = Path(__file__).resolve().parent.parent / "05_your-gpt-from-a-blank-file"


def _load_project_5():
    spec = importlib.util.spec_from_file_location("project_05_build", PROJECT_5 / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["project_05_build"] = module
    spec.loader.exec_module(module)
    return module


p5 = _load_project_5()


def init_weights(model: p5.GPT) -> None:
    """nanoGPT-style init: N(0, 0.02) for linear/embedding; scaled for residual projections."""
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    # Scaled init for residual-path projections.
    n_layers = model.cfg.n_layers
    scale = 0.02 / math.sqrt(2 * n_layers)
    for block in model.blocks:
        torch.nn.init.normal_(block.attn.proj.weight, mean=0.0, std=scale)
        torch.nn.init.normal_(block.mlp.net[2].weight, mean=0.0, std=scale)  # second linear in MLP


def configure_param_groups(model: p5.GPT, weight_decay: float = 0.1) -> list[dict]:
    """Group parameters into decayed and undecayed for AdamW.

    Weights (Linear, Embedding) get weight decay. Biases and LayerNorm params don't.
    Returns groups in the format AdamW expects.
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen_params: set[int] = set()  # ids, for weight-tying dedup

    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if id(param) in seen_params:
                continue
            seen_params.add(id(param))
            if isinstance(module, nn.LayerNorm) or name.endswith("bias"):
                no_decay.append(param)
            else:
                decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train_with_groups(
    model: p5.GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    weight_decay: float = 0.1,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    g = torch.Generator().manual_seed(seed)
    groups = configure_param_groups(model, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(groups, lr=lr)
    train_hist: list[float] = []
    val_hist: list[float] = []
    for step in range(steps):
        x, y = p5.get_batch(train_data, model.cfg.block_size, batch_size, g)
        _, loss = model(x, y)
        optimizer.zero_grad()
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % eval_every == 0 or step == steps - 1:
            train_hist.append(float(loss.item()))
            with torch.no_grad():
                vx, vy = p5.get_batch(val_data, model.cfg.block_size, batch_size, g)
                _, vloss = model(vx, vy)
                val_hist.append(float(vloss.item()))
    return train_hist, val_hist


def write_loss_comparison(
    base_train: list[float],
    base_val: list[float],
    nano_train: list[float],
    nano_val: list[float],
    eval_every: int,
    path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    xs = [i * eval_every for i in range(len(base_train))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(xs, base_train, label="prototype")
    ax1.plot(xs, nano_train, label="nanoGPT-style")
    ax1.set_xlabel("step")
    ax1.set_ylabel("train loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Train loss")
    ax2.plot(xs, base_val, label="prototype")
    ax2.plot(xs, nano_val, label="nanoGPT-style")
    ax2.set_xlabel("step")
    ax2.set_ylabel("val loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Val loss")
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
    )
    args = parser.parse_args()

    if args.steps is None:
        args.steps = 5000 if args.full else 300

    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = p5.DEFAULT_CORPUS
    stoi, _, vocab_size = p5.char_tokenizer(text)
    data = p5.encode(text, stoi)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]

    cfg = p5.GPTConfig(block_size=32, n_layers=4, n_heads=4, d_model=64)
    eval_every = max(1, args.steps // 10)

    # === Prototype (Project 5 defaults: PyTorch default init, uniform weight decay) ===
    torch.manual_seed(args.seed)
    proto = p5.GPT(cfg, vocab_size=vocab_size)
    # Project 5 trained with no weight_decay specified (default 0). Match that.
    proto_train, proto_val = p5.train_gpt(
        proto,
        train_data,
        val_data,
        steps=args.steps,
        batch_size=32,
        lr=3e-3,
        eval_every=eval_every,
        seed=args.seed,
    )

    # === nanoGPT-style (scaled init + parameter groups) ===
    torch.manual_seed(args.seed)
    nano = p5.GPT(cfg, vocab_size=vocab_size)
    init_weights(nano)
    nano_train, nano_val = train_with_groups(
        nano,
        train_data,
        val_data,
        steps=args.steps,
        batch_size=32,
        lr=3e-3,
        eval_every=eval_every,
        weight_decay=0.1,
        seed=args.seed,
    )

    uniform = math.log(vocab_size)
    print(f"\nUniform baseline: {uniform:.4f}")
    print(f"\n{'mode':30s}  {'final train':>14s}  {'final val':>12s}")
    print("-" * 60)
    print(f"{'prototype (P5 defaults)':30s}  {proto_train[-1]:>14.4f}  {proto_val[-1]:>12.4f}")
    print(f"{'nanoGPT-style refinements':30s}  {nano_train[-1]:>14.4f}  {nano_val[-1]:>12.4f}")

    # Sanity: check that param groups split as expected.
    nano_groups = configure_param_groups(nano, weight_decay=0.1)
    n_decay = sum(p.numel() for p in nano_groups[0]["params"])
    n_no_decay = sum(p.numel() for p in nano_groups[1]["params"])
    print(f"\nParameter groups in nanoGPT-style optimizer:")
    print(f"  with weight_decay=0.1:  {n_decay} params")
    print(f"  with weight_decay=0.0:  {n_no_decay} params (biases + LayerNorm)")

    write_loss_comparison(
        proto_train,
        proto_val,
        nano_train,
        nano_val,
        eval_every,
        args.output_dir / "loss_comparison.png",
    )

    log = args.output_dir / "run_log.txt"
    log.write_text(
        "# Project 6 run log\n\n"
        f"uniform baseline           : {uniform:.4f}\n"
        f"prototype (P5 defaults)    : train={proto_train[-1]:.4f}  val={proto_val[-1]:.4f}\n"
        f"nanoGPT-style              : train={nano_train[-1]:.4f}  val={nano_val[-1]:.4f}\n"
        f"\nparameter groups:\n"
        f"  decayed                  : {n_decay}\n"
        f"  not-decayed (bias + LN)  : {n_no_decay}\n"
        "\n"
        "Refinements applied:\n"
        "  1. Init: N(0, 0.02) for Linear + Embedding; residual projections\n"
        "     get std = 0.02 / sqrt(2 * n_layers) to compensate for stacking.\n"
        "  2. AdamW param groups: weights get weight_decay=0.1; biases and\n"
        "     LayerNorm params get weight_decay=0.0.\n",
        encoding="utf-8",
    )
    print(f"\nOutputs written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
