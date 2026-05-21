# Project 20: Autonomous Experimentation

## Hook

Why would anyone let an AI agent edit a training file, run a five-minute experiment, decide whether the change helped, and then keep going without asking permission? That sounds reckless right up until you remember how most model work actually feels. Tweak one line, wait, squint at the metric, forget what changed, try something else, get interrupted, lose the thread.

Project 20 starts from that irritation. The question is not "can an agent invent new science overnight?" The question is much simpler: can you turn experimentation itself into a loop that runs while you sleep, keeps the wins, throws away the failures, and leaves behind a clean record of what happened?

This chapter is also where I have to be honest about my own bias. I have run thousands of experiments through loops like this, on training-run sweeps and on the book itself, and my opinion is sharper than the spec wants. So I will warn you up front.

## The Concept

Think about model research like tuning a race car when every engine check takes five minutes. You can do it by hand. Open the hood, change one thing, run a lap, write down the time, repeat. It works. It scales terribly. You need focus, discipline, and a memory for what you already tried. Then you go home and progress stops.

Autonomous experimentation changes one thing: direct code edits stop being the main act. Instead, you write down the rules of the game for an agent. That rulebook is `program.md`, and it says what file the agent may edit, what metric counts as success, what files are off-limits, how to log results, when to keep a change, when to throw it away, and what to do when a run crashes. You are no longer only programming Python. You are programming a researcher.

The best analogy here is a lab notebook plus a junior engineer plus a strict supervisor, all collapsed into one loop. `results.tsv` is the notebook. The coding agent is the junior engineer. Git is the strict supervisor. Git matters more than people think in this project. It is more than version control. It is experiment memory and a safety rail. A kept experiment advances the branch. A failed experiment gets erased by resetting back to the last good commit. The branch becomes a climbing path through better and better ideas.

Not every idea survives, and that is the point. Most experiments should die. If everything survives, your metric is weak or your discipline is gone.

The important mental flip is this. The project is not "AI does research for you." It is "AI runs a bounded search loop inside a sandbox you defined." That sounds less magical, and good. It is also more useful.

Strong opinion, since I promised one. The phrase "AI does research" is the worst marketing of the last two years. Research is a long horizon of choosing which question to ask. A loop like this does not do that. It runs a bounded search inside a question you already chose. If you confuse those two things, you will trust the loop with judgment calls it cannot make. If you keep them separate, the loop is one of the most useful tools in this book.

The repo [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) keeps the setup small on purpose. According to the current README, only three files really matter: `prepare.py`, `train.py`, and `program.md`. `prepare.py` handles one-time data prep and evaluation and you do not touch it. `train.py` contains the model and training loop and the agent edits this. `program.md` tells the agent how to behave and you edit this. That division is the lesson. Earlier projects built model pieces; this one builds a process.

And the process has a metric: `val_bpb`, validation bits per byte. In plain English: how surprised is the model, on held-out text, per byte of data? Lower is better. If the number goes down, the model predicts better.

This is where the repo makes a sharp engineering choice. It fixes the training budget to five minutes of wall-clock training time, excluding startup and compilation overhead, then compares experiments by the same metric under the same time budget. That choice turns an argument into a game you can actually score. Without a fixed time budget, an agent could "improve" the model by making it slower and just training longer. With a fixed time budget, the agent has to make tradeoffs. It can pick a smaller model that trains more steps in five minutes, or a larger model that learns more per step but runs fewer steps. It can change batch size, learning rate, normalization, optimizer details, attention pattern, initialization, or architecture. But it cannot cheat the clock.

That is why this project belongs right after **Project 19: Scaling Laws**. **Project 19: Scaling Laws** teaches that compute allocation is a contract. This one teaches that search over that contract can itself be automated. You are no longer asking only "What model should I train?" The question has shifted to "What system should search the space of models for me?"

## Why It Matters

Without a loop like this, experimentation has four common failure modes. First, you stop too early. A human researcher hits one or two bad runs and quietly narrows the search space to "safe" ideas, so the weird ideas never get tried because they are annoying to test. Second, you forget context. You remember that "normalization before attention seemed bad" but you no longer remember what else changed in that run, so you can't tell whether it was really the normalization placement or also the batch size and learning rate. Third, you lose negative knowledge. A failed run still teaches something, whether that a particular width causes out-of-memory errors or that an aggressive learning rate gives a short burst of progress and then turns the loss to `NaN`, and if you do not log failures you end up paying to rediscover them. Fourth, your sleep schedule becomes the bottleneck, which is ridiculous once you see it clearly. The GPU can work all night, your process cannot.

All four of these are failure modes I have actually committed. I caught myself forgetting "what else changed in that run" once while writing this book, comparing two notebook experiments two weeks apart and realizing I had silently changed the data shuffler in between. The git history saved me. The notebook did not.

Autonomous experimentation fixes those four things by turning every trial into a small contract: change one thing or a small cluster of things, commit it, run it, measure it, record it, keep it if it helps, and revert it if it does not. That sounds almost insultingly simple, which is why it works.

There is another reason this matters. Most explanations of AI agents make them sound like mystical problem-solvers. This project cuts through that. The agent is only useful because the environment is tight. One editable file, one metric, one fixed run length, one logging format, one revert rule. That is what makes the agent productive instead of chaotic. Hand an agent an unbounded codebase and say "improve my model," and you do not have a research system. You have a liability. Hand it a small search space with a hard metric and a rollback rule, and now you have something real.

This project also teaches a bigger lesson: many "AI workflows" are just optimization loops in disguise. You define a search space, an objective, a safety policy, and a memory of past attempts. That pattern shows up again in fine-tuning, hyperparameter search, evaluation harnesses, inference tuning, and product prompt iteration. Once you see that, "agentic AI" stops sounding like a marketing category and becomes controlled trial-and-error with better tooling.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/20_autonomous-experimentation/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/20_autonomous-experimentation/build.py --full

# The BREAK IT experiment:
python projects/20_autonomous-experimentation/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 20 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
