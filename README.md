# Under the Hood

### Build Every Layer of a Large Language Model from Scratch

> *A practical manual for understanding how modern language models are built, where they fail, and how to reason about their behavior like an engineer instead of a spectator.*

**By Ramchand Kumaresan**

**Build it. Break it. Measure it.**

---

## What This Book Is

Most LLM books teach you to *use* models. This one teaches you to *build* them — every layer, every optimizer step, every cache, every quantization scheme — and then to deliberately break each piece so you understand why it exists in the first place.

It is a workshop in book form. Twenty projects, ~470 pages, one tight spiral that takes you from a single autograd scalar all the way to fusing independently trained specialists into a routed system. No black boxes. No "import library, call method." You write the code, you run it, you break it, you measure what broke.

If you have ever read a transformer paper and felt that the diagram and the code were in two different universes — this book closes that gap.

---

## Who This Is For

- **Engineers** who can ship software but feel like LLM internals are wizardry happening behind an API.
- **Researchers** who want a working mental model of the full stack before reading their hundredth attention paper.
- **Students** who learned PyTorch from notebooks and want to know what a training run *actually* does.
- **Practitioners** tuning, deploying, or fine-tuning models who keep hitting failure modes they cannot explain.

You should be comfortable with Python and have seen a tensor before. Everything else is built up from scratch.

---

## The Method: Build → Break → Measure

Each project has the same disciplined rhythm:

1. **Hook** — the question the project answers.
2. **The Concept** — the idea in plain English, before any code.
3. **Why It Matters** — what fails in the real world without this piece.
4. **The Build** — step-by-step implementation, written so shapes stay honest and code stays readable.
5. **Break It** — a deliberate sabotage of the system, with predictions and observations, so you learn *why* the working version works.
6. **Optional Homework** — full lab, proxy lab (for limited hardware), and result-guided versions.
7. **Questions To Answer** — the small set of things you should be able to defend after finishing.
8. **Go Further** — research anchors and next directions.
9. **What You Now Know** — the explicit deltas in your mental model.

Reading the book without breaking the code is half the experience. The breaks are where the lessons actually live.

---

## The Twenty Projects

### Foundations
1. **The Learning Machine** — a scalar autograd engine, a neuron, a tiny MLP, and a training loop you wrote yourself.
2. **Predicting the Next Character** — bigram counts → learned embeddings → temperature → negative log-likelihood, end to end.
3. **Building a Tokenizer** — byte-pair encoding from scratch; vocabulary size as a tunable knob with real tradeoffs.
4. **Attention from Scratch** — Q/K/V, scaled dot-product, masking, multi-head, entropy per head, on real text.

### Building a GPT
5. **Your GPT from a Blank File** — the smallest complete system: tokenizer → batches → transformer → schedule → checkpoints → samples.
6. **From Prototype to nanoGPT** — side-by-side with Karpathy's reference; what production-shape code looks like and why.
7. **The Details That Matter** — LayerNorm vs RMSNorm, GELU/SwiGLU, learned positional embeddings vs RoPE, instrumentation.

### Training at Scale
8. **Pretraining on the Real Web** — FineWeb-EDU via nanochat, mixed precision, gradient accumulation, LR sweeps, val-bpb.
9. **Fast Inference: The KV Cache** — naive autoregressive generation → cached → benchmarked → sliding window.
10. **Grouped Query Attention** — MHA → GQA → MQA, with KV-cache shape and benchmark tradeoffs made explicit.
11. **Mixture of Experts** — router, top-k, expert utilization, load balancing loss, active vs total parameters.
12. **Scaling Laws** — depth sweeps, compute-optimal training, fitting the law, extrapolating to a target loss.
13. **Autonomous Experimentation** — pointing an agent at a `program.md` and reading the git history as a research diary.

### Aligning and Deploying
14. **Fine-Tuning and Instruction Tuning** — SFT, conversational tuning, LoRA vs full FT, catastrophic forgetting on purpose.
15. **Reward Models and RLHF** — reward dataset, RM training, GRPO, KL leash, what reward hacking actually looks like.
16. **Quantization and Deployment** — FP32 → INT8 → INT4 → GGUF → llama.cpp; where quality breaks before grammar does.

### Modular Composition
17. **Layer Freezing and Transfer** — freeze ratios, CKA, training speed vs forgetting vs domain gain.
18. **Fusing Independently Trained Specialists** — shared base, specialist heads, router on a mixed validation set, oracle gap.
19. **The Interface Specification** — a machine-readable contract for modular specialists; loud failures vs silent ones.
20. **Incremental Assembly** — order independence, scaling limits, and what "modular" actually has to mean to be real.

---

## What You Walk Away With

By the end you will have:

- A scalar autograd engine you can defend line by line.
- A character-level model, a BPE tokenizer, and a multi-head attention block — all written by hand.
- A small GPT trained end-to-end on real web data, instrumented and checkpointed.
- A KV cache, a GQA implementation, and a benchmarked MoE layer.
- A reward model and a working GRPO loop on GSM8K.
- A quantized model exported to GGUF and running in `llama.cpp`.
- A fused multi-specialist system with a typed interface contract.
- And — more importantly — a debugging instinct. You will read a training curve and *know* what is wrong.

---

## How to Use This Book

- **Read in order.** Each project's "Starting Point" assumes the previous one's code is in your hands.
- **Type the code yourself.** Not because copy-paste is forbidden — because the goal is to internalize shapes, not to produce artifacts.
- **Run every Break It.** Predict what will happen first, then run it, then write down what actually happened. The gap between prediction and observation is the lesson.
- **Do at least the proxy lab.** Even if you do not have an H100, the result-guided versions are designed so you still get the structural insight.
- **Keep a notebook.** Each project asks you to record specific metrics. The notebook *is* the textbook by the end.

---

## Hardware Notes

The book is honest about what each project needs:

| Project Range | Realistic Hardware |
|---|---|
| Projects 1–7 | CPU or any laptop GPU |
| Projects 8–13 | One consumer GPU (8–24 GB) or short cloud rental |
| Projects 14–16 | One consumer GPU; LoRA paths designed for tight VRAM |
| Projects 17–20 | One GPU for full labs; proxy versions documented for all |

Every project has a **proxy lab** for limited hardware and a **result-guided version** for readers who cannot run anything at all. The pedagogy works at all three tiers.

---

## Philosophy

> *Build it. Break it. Measure it.*

This is not a tagline. It is the entire epistemology of the book.

- **Build it** because reading is not understanding. Understanding is being able to reproduce.
- **Break it** because a working system tells you what it does. A broken system tells you why it has to be that way.
- **Measure it** because intuition without numbers is the most expensive failure mode in machine learning.

Modern LLMs are not magic. They are a stack of carefully-tuned engineering decisions, each one earned by a specific failure that someone, somewhere, had to debug at 2am. This book makes you the person who debugged it.

---

## About the Author

**Ramchand Kumaresan** writes systems-level engineering manuals for AI practitioners. His work focuses on the parts of the stack that are usually skipped: the failure modes, the diagnostics, the parts of the codebase that exist as scar tissue from a real outage. *Under the Hood* is the first book in a series on building production-grade machine learning systems from first principles.

---

## License & Distribution

This repository hosts the README and project metadata for the book *Under the Hood: Build Every Layer of a Large Language Model from Scratch*.

The PDF is distributed separately. For access, reach out to the author.

---

*If you finish this book and can still be surprised by the behavior of an LLM, you did not break enough of it.*
