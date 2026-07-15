# Project 31: Diffusion and Non-Autoregressive Decoding

## Hook

Every model in this book so far has written the same way. Left to right. One token, then the next token, then the next, each one conditioned on everything already placed and never revised. **Project 5: Your GPT From A Blank File** did it. **Project 30: Non-Transformer Architectures (Mamba, RWKV)** changed how the model stored its history, but not the order in which it spoke. The pen only ever moves forward.

So here is the question this chapter opens with. Is left-to-right the only way to generate text? What would it take to write a sentence the way a person fills in a crossword, committing the words they are sure of first and letting those pin down the rest?

This chapter builds that alternative decoder by hand, on a small masked language model, on a laptop CPU if that is all you have. You will watch a sentence resolve out of a row of blanks, and you will find out exactly where the trick pays off and where it does not.

## The Concept

Picture a crossword grid, empty. Every cell is a blank.

An autoregressive model solves it like someone forced to fill the cells in reading order, top-left to bottom-right. They write a final letter in each cell before moving to the next. No erasing, ever. That works, but it is a strange rule to impose on a crossword. The natural way to solve one is different. You scan all the clues. You fill in the three or four answers you are certain about. Those committed letters become constraints, and the crossings you were unsure about turn solvable. You do several passes. Each pass, you write down what you are now confident of and leave the doubtful cells blank for the next pass.

That second solver is a **diffusion language model**, or dLLM. The rest of this chapter is a careful account of how it turns a grid of blanks into a sentence. And why that buys speed, at a cost you can measure.

Start with the words. An **autoregressive** model, abbreviated AR, generates one token at a time in a fixed left-to-right order, each token conditioned only on the tokens before it. That is every GPT in this book. **Non-autoregressive** means the model does not commit to that order: it can produce or revise tokens at many positions in the same step. A dLLM is non-autoregressive.

Now the mechanism, in the exact form the research uses. The dLLMs this chapter is about (LLaDA, Mercury, Gemini Diffusion) are **masked**, or **discrete**, diffusion models. Here is the one sentence to hold onto before any code. In a text diffusion model the "noise" is a fully-masked sequence, not continuous random values. Generation runs by iteratively unmasking that sequence. Say it that way and you will not fall into the common wrong picture. Image diffusion models start from a field of random Gaussian pixels and denoise them. Text dLLMs do not. Their starting state is a row of blanks. A blank is a special vocabulary symbol, the **mask token**, that means "no token decided here yet." The LLaDA paper (arXiv:2502.09992, from Nie and colleagues, ICLR 2025) builds exactly this: an all-masked sequence and a reverse process that fills it in.

The thing doing the filling is a **mask predictor**. That is a Transformer with no causal mask, so every position can read every other position in both directions. Call that **bidirectional** attention: unlike the causal attention of a GPT, where a token sees only its past, here a masked cell in the middle of the sequence sees the whole sequence at once, left and right. You built the bidirectional version already, without the name, in the vision encoder of **Project 29: Multimodal: A Tiny Vision-Language Model**. Same idea, different data.

A **denoising step** is one forward pass of the mask predictor followed by a decision about which blanks to fill. The model reads the current partly-filled sequence. It predicts a probability distribution over the vocabulary for every masked position at once. And it reports a **confidence** for each: how sure it is of its best guess there, read straight off the top probability. Then comes the rule that makes the whole thing work. Keep the high-confidence guesses. Commit them as real tokens. Leave the low-confidence positions blank, or blank them back out if an earlier step filled them badly. That last move has a name, **remasking**: a position that was written can be returned to the mask state on a later step, which is how the model revises. Run this for **T** steps, from an all-masked sequence toward a fully-filled one, and the sequence resolves **coarse-to-fine**: the easy, high-confidence words first, the words that depend on them last.

Researchers have a formal name for this setup: absorbing-state diffusion. The mask token is the absorbing state. A forward process corrupts real text by sending positions to the mask, and once a position is masked it stays masked until the reverse process fills it. So the entire model reduces to one learned skill. Given a partly-masked sequence, guess the masked tokens. That is all it ever does. This is why an ordinary masked language model can stand in for the decode loop at all: it already has that skill, learned at a low masking rate. A dLLM has the same skill at every masking rate, all the way up to the all-mask start.

Map the analogy onto the tensors before the build, so the crossword becomes concrete. The grid is the token sequence. A blank cell is a position holding the mask id. A pencilled-in answer is a position holding a real vocabulary id. The solver's certainty is the top softmax probability at that position. A pass over the grid is one forward call of the mask predictor. Filling the sure clues is committing the highest-confidence positions. Rubbing out a doubtful cell is remasking. Every noun in the analogy is a value you can print in the build below.

![Figure 31.1. Left-to-right versus iterative unmasking. Top: an autoregressive model commits one token per step in reading order and never revises. Bottom: a masked diffusion model starts from all-mask, predicts every position in parallel, commits the most confident, remasks the rest, and repeats over T denoising steps.](figures/fig_ar_vs_diffusion.png)

Contrast the two orders plainly. The AR model asks, at every step, one question: given everything so far, what is the single next token? The dLLM asks a different question at every step: given everything decided anywhere in the sequence, which blanks are now sure enough to fill? The AR model's answer is one token. The dLLM's answer is a batch of tokens, chosen by confidence rather than by position. That is the entire architectural difference, and everything else in this chapter follows from it.

One honest word before the payoff, because it is the part marketing tends to drop. Filling many positions at once is fast, but it is not free of trouble. When the model commits several blanks in a single step, it decides each of them from the same context, without seeing the others it is committing alongside. If two of those positions needed to agree with each other, nothing forced them to. Fewer steps means more positions decided in parallel per step, which means more of these missed agreements. That tension is the whole game. It is what BREAK IT will make concrete.

## Why It Matters

The reason anyone builds these is throughput. An autoregressive model's speed has a hard floor. It runs one forward pass per token, in order. So a 200-token answer is 200 sequential passes, no matter how much hardware you own. A dLLM breaks that floor by deciding many tokens per pass, and the reported numbers are large.

State them with their conditions attached, because the conditions are the whole story. Inception Labs reports its Mercury Coder models running at over 1000 tokens per second on NVIDIA H100 GPUs: Mercury Coder Mini at 1109 tok/s and Mercury Coder Small at 737 tok/s, per the Mercury paper (arXiv:2506.17298) and the company's launch blog. Those are code-optimized models, single-request throughput, on commodity H100s rather than custom silicon, and they are the vendor's own benchmark. Inception Labs positions this as up to roughly 10x faster than speed-optimized frontier autoregressive models, with the blog phrasing it 5 to 10x. That is measured against AR baselines in the 200-tok/s class. It is not a quality-matched comparison. Google DeepMind reports a second system, Gemini Diffusion, at about 1479 tok/s, framed as the quality of Gemini 2.0 Flash-Lite at roughly 5x the speed. Read the label on that one too. DeepMind calls it an experimental demo, not a shipped product. The speed figure excludes a fixed overhead in their own chart. When a 200-token answer arrives in a fifth of the time, an interactive coding assistant stops feeling like a typewriter and starts feeling like pasting the whole answer at once.

Now the other half, stated just as plainly, because a chapter that only sold the speed would be lying by omission. Diffusion LLMs do not beat autoregressive models on reasoning. On DeepMind's own evaluation table, Gemini Diffusion scores 15.0% on BIG-Bench Extra Hard against 21.0% for Gemini 2.0 Flash-Lite. A clear reasoning gap, not parity. The Efficient-DLM paper (arXiv:2512.14067) notes that diffusion models' learning efficiency lags autoregressive models when trained from scratch. Where dLLMs are competitive is code and short-form generation. On the same DeepMind table, Gemini Diffusion posts 89.6% on HumanEval against Flash-Lite's 90.2%. On LiveCodeBench it is 30.9% against 28.5%. Even there, close, not a rout. The open research anchor is LLaDA, an 8B diffusion model trained from scratch. The paper shows it competitive with a comparable autoregressive baseline (LLaMA3 8B) on in-context learning and instruction following. It also mitigates the "reversal curse," even beating GPT-4o on a reverse-poem-completion task. Competitive with. Not uniformly better. The picture is a speed-versus-capability frontier, strongest on code and short answers, still behind on hard reasoning and long-form coherence.

The production status matters when you decide whether to reach for one. As of mid-2026 the three anchors sit at three different stages. Mercury is shipped. Inception Labs offers a playground, an API, and on-prem deployment. Gemini Diffusion is an experimental demo from Google DeepMind, not a generally available product. LLaDA is open research with public weights on Hugging Face. So the one you can pull down and run today, weights and all, is LLaDA. The one you can call in production is Mercury. The build below imitates the decode loop both of them use, on hardware you already own.

There is a connection back into the book that is easy to miss, and worth pinning down because it prevents a real confusion. The phrase "block diffusion" already appeared in this book, in a completely different job. **Project 14: Speculative Decoding** used a lightweight block-diffusion model as a *drafter*. The DFlash method (arXiv:2602.06036) drafts a whole block of guess tokens in one non-causal forward pass. Then a large autoregressive model verifies every one of them and throws out the wrong guesses. The DFlash paper reports up to about 6x lossless acceleration across models (around 4.9x average on Qwen3-8B, greedy, temperature 0). That is diffusion used as a fast intern whose work a left-to-right model checks. This chapter is the other use entirely: here the diffusion model *is* the decoder, and nothing downstream re-does its work left to right. Same word, "diffusion," two roles. In **Project 14** it is a draft mechanism inside AR serving. Here it is the full non-autoregressive model. Both are masked/discrete diffusion, and in neither case is the "noise" continuous Gaussian noise.

The serving side connects too. Deciding many tokens per pass raises GPU utilization, but a long context makes the key-value cache from **Project 13: Fast Inference: The KV Cache** grow until it dominates memory. The HERALD system (arXiv:2606.21633) targets exactly that for block-diffusion serving. It reports up to 1.59x lower per-block latency and 2.47x higher throughput than a GPU-only baseline, at near-lossless accuracy with a 5 to 10% KV budget. Those figures were measured across three block-diffusion LLMs on five long-context tasks, built on SGLang. That is the same production-serving concern as **Project 17: Production Serving: Continuous Batching and PagedAttention**, aimed at a non-autoregressive decoder.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/31_diffusion-and-non-autoregressive-decoding/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/31_diffusion-and-non-autoregressive-decoding/build.py --full

# The BREAK IT experiment:
python projects/31_diffusion-and-non-autoregressive-decoding/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 31 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
