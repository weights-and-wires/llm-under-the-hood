# Project 19: Scaling Laws

## Hook

If a bigger model is better, why not just keep the small one and train it longer? That question sounds reasonable right up until you watch a tiny model chew through the same data forever, improve by crumbs, and still lose to a larger model that trained for less time. That feels backwards the first time you see it.

This chapter is about why it is not backwards at all. It is budgeting.

I have run more model-size sweeps than I care to count. A long set of cooperative LoRA-fusion experiments alone covered four families from a few hundred million up to several billion parameters across thousands of runs. The single most expensive lesson of all of that work was not which model wins. It was how many of the runs were sized wrong on day one, which I only knew after I had paid for the result.

## The Concept

**Project 18: Mixture Of Experts** asked a routing question: how much of the model wakes up for each token? This chapter asks a harsher question: how much model should exist in the first place?

Imagine you run a newsroom with a fixed payroll budget. You can spend that budget on two things: hiring more reporters, or giving the reporters more time in the field. A tiny newsroom with endless field time has a problem. The same few reporters keep coming back to the office with more notes than the editors can meaningfully absorb. They are not learning nothing; they are learning slowly, and the bottleneck is now the size of the staff. A giant newsroom with no field time has the opposite problem: you hired a lot of people, but most of them never got enough exposure to learn the beat. Models work the same way.

A model has parameters. Each one is an adjustable number inside the network, a piece of storage for learned patterns. Training data arrives as tokens, the basic unit the model reads and predicts: word fragments, punctuation, whitespace, and so on. Your compute budget has to be split between parameter count (how much pattern-storage you built) and training tokens (how much experience you gave that storage). Too few parameters for the token budget and the model becomes a tiny notebook you keep cramming more notes into. Too many parameters for the token budget and you bought a library but only stocked one shelf.

Scaling laws are the measured relationship between these knobs and the loss, one number that tells you how wrong the model is, where lower is better. The first useful mental move is to stop asking "Is this model big?" and start asking "Is this model sized correctly for the compute I can afford?" That is the real question.

When people say "scaling law," they often mean one of two related patterns. The first is simple: if you increase model size, loss tends to go down in a smooth, predictable way. The second is the one that changes engineering decisions: for a fixed compute budget, there is a better balance between model size and training duration than "small model, train forever." That second pattern is the Chinchilla result.

The blunt version is that undertrained large models waste capacity, overtrained small models waste compute, and there is a sweet spot between them. The famous rule of thumb from Chinchilla is about 8 training tokens per parameter for compute-optimal training. That does not mean 8 is sacred for every setup forever. It means there is a ratio, and if you are far from it, you are probably wasting money.

I treat the number as a north star, not a law. On training runs at scale the practical compute-optimal ratio drifted depending on data quality, optimizer warmup, and whether we were doing cooperative LoRA fusion or training from a clean init. Treat 8 the way a chef treats a recipe ratio: useful for planning, embarrassing if you follow it without tasting.

This is why nanochat parameterizes so much by depth. Depth is the number of transformer blocks, the stacked layers of the model. In nanochat's teaching arc, depth is the master size knob because changing it changes capacity in a smooth, easy-to-sweep way, and you can derive or set other choices around it. That makes scaling experiments much easier to reason about than changing ten knobs at once.

Picture a curve where each extra chunk of parameters buys you less improvement than the last one. The first jump from tiny to small helps a lot. The jump from large to larger still helps, but less. That is diminishing returns, and it explains why simply continuing to scale is not a free lunch.

Now you have earned the compact version. Papers often write a scaling fit as:

```text
L = a * P^(-b)
```

I am using `P` for parameter count here because this book already uses `N` for depth. What each symbol means is straightforward. `L` is loss. `P` is the number of parameters. `a` is a fitted constant that sets the height of the curve, and `b` is a fitted constant that tells you how quickly loss improves as you scale up.

Read it in English like this: loss falls like a power of parameter count. That sounds abstract until you plot it on log axes. Take logs of both sides:

```text
log L = log a - b log P
```

Now it becomes a line. That is why people love log-log plots for scaling laws. Curved intuition turns into a straight-ish line you can fit.

None of this says the world is exactly one neat equation. Real runs have noise, data quality matters, architecture details matter, and optimization matters. There is often an irreducible floor below which your simple fit stops being honest. Even a rough fit is enough to do something practical: plan before you spend.

The first time I trusted a scaling fit and predicted a target size from small runs, the prediction was, in fact, within a depth of where the empirical run landed. That surprised me more than I would like to admit. The natural prior is that fitting a curve and plugging in a target loss should give a number that is a factor of two wrong in either direction. Figure 12.1 shows exactly why the log-log view is the right lens for this kind of planning.

![Figure 12.1. Scaling-law curves look curved in raw coordinates but close to linear on log-log axes, which is why they are useful for forecasting model size and token budgets.](figures/fig_scaling_laws_loglog.png)

## Why It Matters

Without scaling laws, model planning turns into cargo cult engineering. You hear "bigger is better," so you make the model bigger. You hear "more data is better," so you train longer. You hear "GPT-2 had X parameters," so you aim at that number because it sounds familiar. That is not planning. That is guessing with a GPU bill attached.

I will be plainer than I usually am here. Most beginner explanations of scaling fall apart at exactly the moment a reader has to make a real budget decision, because the explanations stop at "bigger models do better" and never connect that claim to a dollar number on a cloud invoice. The Chinchilla result is one of the few places in this field where the math actually tells you what to spend.

Scaling laws give you three things. First, they tell you when "train longer" has stopped being a smart answer. This matters because overtraining a small model feels productive: the loss still goes down, the run is still alive, the logs still move. A moving loss curve is not the same as good compute allocation. Second, they tell you how to compare runs fairly. "Same wall-clock time" and "same compute budget" are not the same comparison. A smaller model usually processes more tokens per second, so if you train several models for the same number of hours, the small one sees more data. That is one kind of comparison and it is useful, but it does not answer the compute-optimal question. To answer that, you need to account for both size and duration. Third, scaling laws let you build a cost calculator before you launch the expensive run. That is what labs do, not because they are romantic about curves, but because they do not want to waste weeks on a model that was mis-sized on day one.

This chapter is also the answer to a common misconception: "Bigger is always better."

No. Bigger and trained correctly is better. Bigger and starved is wasteful. Smaller and trained forever is also wasteful. The balance is the lesson.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/19_scaling-laws/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/19_scaling-laws/build.py --full

# The BREAK IT experiment:
python projects/19_scaling-laws/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 19 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
