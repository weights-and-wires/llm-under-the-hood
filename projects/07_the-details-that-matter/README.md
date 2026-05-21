# Project 7: The Details That Matter

> RMSNorm instead of LayerNorm. SwiGLU instead of GELU. The two architectural swaps that appear in Llama, Mistral, and most modern open-weight LLMs. Train all four combinations side-by-side and look at the deltas.

## Hook

By the time you have read three different open-weight LLM codebases you start noticing that they all dropped vanilla LayerNorm + GELU somewhere between 2022 and 2024. RMSNorm and SwiGLU are the standard replacements. This project implements both, trains all four `{LN, RMS} × {GELU, SwiGLU}` combinations against the same data, and reports the comparison.

## The Concept

- **RMSNorm.** LayerNorm computes mean and variance, then normalizes. RMSNorm drops the mean step and only divides by root-mean-square: `x / sqrt(mean(x²) + eps)`. Same effect on signal magnitude; about half the work.
- **SwiGLU.** Standard MLP: `x → Linear → GELU → Linear`. SwiGLU adds a gating term: `(silu(x @ W_gate) * (x @ W_up)) @ W_down`. The gating multiplies the projected representation by a learned mask — features get scaled by their own activation, which makes the MLP more expressive without changing parameter count much (we set hidden_mult=8/3 to keep counts roughly equal to the GELU MLP).

## Why It Matters

These are not algorithmic breakthroughs. They are incremental refinements that win on the margins. Reading reference implementations of modern models without knowing what's underneath these names is reading code with two unexplained imports.

---

## What Got Built

A single script that defines `RMSNorm`, `SwiGLU`, and a `ModernGPT` with configurable norm + MLP. Trains all four combinations and reports.

### Files in this folder

| File | What it is |
|------|------------|
| [`build.py`](build.py) | `RMSNorm`, `SwiGLU`, `ModernGPT` with config switches; trains 4 variants |
| `step_*.py` | The book's code blocks, extracted step-by-step. Reference material. |
| `tests/test_unit.py` | 7 tests: RMSNorm produces unit-RMS output, SwiGLU shape, ModernGPT forward, all 4 variants train |

### How to run

```bash
python build.py --tiny      # 200 steps each = 4 quick trainings, ~30s on CPU
python build.py --full      # 2000 steps each
pytest projects/07_the-details-that-matter/
```

---

## Outputs (from `python build.py --tiny`)

200 training steps × 4 variants × ~25k params each:

```
norm    mlp           params     train       val
--------------------------------------------------
ln      gelu          105216    1.5537    2.6379
rms     gelu          104896    1.5506    2.5884
ln      swiglu        104320    1.5969    2.5978
rms     swiglu        104000    1.6135    2.5991

Uniform baseline: 3.8712
```

### Reading the table

- All four variants train comfortably below uniform (3.87).
- RMS + GELU edges out LN + GELU by 0.05 nats on val. RMSNorm is mostly a "same quality, less work" swap.
- SwiGLU variants slightly raise train loss but reduce val loss vs. plain LN+GELU — small regularizing effect at this scale.
- The deltas here are **modest** (~0.05 nats). On a tiny 1.8 KB corpus you should not expect dramatic differences. At pretraining scale, however, RMSNorm reliably accelerates training (fewer FLOPs per step) and SwiGLU compounds the per-param expressivity gains across millions of steps.

The lesson: these refinements are not pedagogical noise. They are small in isolation but compound at scale. The reason every recent open-weight model ships with RMSNorm + SwiGLU is not that any one swap is large — it's that they stack.

---

## No separate BREAK IT

The four-variant comparison IS the experiment. If you want a "broken" version, run with `norm_type="ln", mlp_type="gelu"` (the older recipe) as a baseline and compare. The cleaner "what fails" exercise is to disable LayerNorm entirely (zero the gain parameter, freeze it) — left as an exercise.

---

## Read in the book

This project is Chapter 7 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.

Read the chapter for: the RoPE (rotary positional encoding) derivation that we skipped in this minimal build (it deserves its own treatment), instrumentation diagnostics for inspecting per-layer activations, and the long-form case study on why these particular details were the ones that propagated across modern LLM architectures.
