# Project 1: The Learning Machine

## Hook

Why does `.backward()` feel like a spell? You write a few lines of code, call one method, and somehow every weight in a network learns how to change itself. If you are new to machine learning, this is the first insult the field throws at you: the most important step is hidden behind a single line. I had been calling `.backward()` for years before I sat down to write a toy version from scratch, and the embarrassing truth is that until I did, I could not have told you precisely what was happening inside it. This chapter fixes that. You are going to build the thing that computes those gradients yourself, break one tiny piece of it, and watch learning stop cold.

## The Concept

A neural network learns in three moves: it makes a guess, you measure how wrong that guess is, and then you figure out which internal numbers caused the error and by how much. That third step is the whole story.

People give it intimidating names (automatic differentiation, reverse-mode autodiff, backpropagation), but strip away the names and it becomes plain. The way I actually think about it: every number a network produces has an ancestry, and learning is the act of walking that ancestry backward and asking each ancestor "how much of this mess was your fault?" The final number in the chain is the loss, the score that says how wrong the model is. The loss only exists because some intermediate number contributed to it, which only exists because some earlier number contributed to that. If you want to know why the loss is so high, you walk backward through every step that fed it.

To make it more concrete: think of a recipe. You bake a cake, it tastes bad. Too salty, too dense, not sweet enough. You want to know what to change next time. Was it too much salt? Not enough sugar? Too much flour? Did two ingredients interact in a way that made the result worse? A neural network does the same thing with numbers. It combines inputs through additions, multiplications, nonlinear squashing functions like `tanh`, and many other steps, and at the end it gets one score: the loss. Backpropagation is blame-assignment through that recipe: which numbers made the loss worse, and if I nudge this number slightly upward, does the loss go up or down?

I started calling it blame-assignment after a debugging session where I had a model whose loss refused to fall, and I realized I had been mentally treating the gradient as a "direction" rather than as an accusation against a specific parameter. Once I started reading `p.grad` as "the responsibility this parameter bears for the current loss," the diagnostics got faster. Weights with a near-zero `grad` were suspects: either they were doing nothing useful, or the path that should have carried their blame was broken.

That sensitivity is the **gradient**: a number telling you which direction to nudge a parameter to reduce the loss, and by how much. The image I keep in my head is that weights are ingredients the recipe never measures exactly, and gradients are the chef tasting the result and silently adjusting tomorrow's quantities by hand. The chef does not redesign the recipe, just nudges. Thousands of those nudges in a row are what training is.

Automatic differentiation means the computer tracks the recipe as it runs. The answer alone is not enough. It keeps the whole trail. Every time you add two values, multiply them, or pass one through `tanh`, the system records what operation happened, which inputs created the output, and how to push blame backward through that operation later. That recorded trail is the **computational graph**, illustrated in Figure 1.1.

![Figure 1.1. A computational graph records both the forward recipe and the backward blame path: each value knows where it came from and how to pass gradient signal back to its parents.](figures/fig_computational_graph_backprop.png)

You can picture it as a family tree of numbers. Leaves at the bottom are raw inputs and trainable parameters. Internal nodes are results of operations like `a * b` or `a + b`. The root at the top is the final loss. The forward pass builds this graph while computing the output, and the backward pass walks that graph in reverse and assigns blame all the way down.

Here is the part most explanations skip: there is no magic leap from loss to all gradients. The system applies one local rule at a time. For addition, if `c = a + b`, then a change in `c` passes straight back to both `a` and `b`, because increasing either input by 1 increases the output by 1. For multiplication, if `c = a * b`, then a change in `c` passes back to `a` scaled by `b`, and back to `b` scaled by `a`. That is just the chain rule from calculus. Each step tells the previous step how much responsibility it had.

I expected the chain rule to feel deep once I built the engine. It did not. It felt mechanical, almost boring: a small local rule called over and over against a list of nodes. That itself was the lesson I needed. The mystique is in the scale of the graph, not in the math at each node.

Let's earn the first equation. Suppose `L` is the final loss, `c` is some intermediate value, and `a` and `b` created `c` through `c = a * b`. We want to know: how much does changing `a` change the loss? We already know two things: how much changing `c` changes the loss (`dL/dc`), and how much changing `a` changes `c` (`dc/da = b`). Put them together:

```text
dL/da = (dL/dc) * (dc/da) = (dL/dc) * b
```

Read it in English: the effect of `a` on the loss equals the effect of `c` on the loss times the effect of `a` on `c`. That is the whole machine, for multiplication and for everything else. Every deep learning framework you use is doing this over and over again, at scale, through giant graphs made of millions or billions of operations. This chapter starts with the toy version because the toy version is the real version, only small enough to see.

Honest opinion, after reading more beginner backprop explanations than I care to count: most of them are terrible, and they fail in the same way. They reach for Jacobians and matrix calculus before the reader has watched a single scalar gradient flow backward through a graph drawn by hand. Build the toy graph first. The Jacobians can wait. If you cannot say which arrow in the graph the gradient is flowing along, the matrix form will not help. It will just hide the confusion under heavier notation.

## Why It Matters

If you do not understand automatic differentiation, every other part of machine learning stays foggy. You can still call APIs, fine-tune models, and read benchmark tables, but you cannot reason from first principles about what training is actually doing. You cannot answer questions like why training stopped improving, why one missing gradient makes the model freeze, or why a bad initialization or saturated activation kills learning entirely.

Every modern model learns through the same loop: compute a prediction, compute a loss, compute gradients, update parameters. Large language models do not escape this. They just repeat it at a scale that hides the mechanism behind more code, more hardware, and more abstractions. Across a long set of fine-tuning experiments I ran on models from a few hundred million to several billion parameters, the failures that hurt most were never the loud ones. The loud ones crash and you fix them. The painful ones were runs where the loss flattened, the program ran fine, and a single broken path in the graph had quietly turned a slab of parameters into dead weight.

If the gradients are wrong, the model does not "kind of learn badly." It learns nonsense. If one path in the graph does not pass blame backward, the parameters behind that path go dark. They stop receiving useful signals. You can have a fully valid program, no crashes, no warnings, and still get a dead learning system. That is what we are going to build toward, and then deliberately break.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/01_the-learning-machine/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/01_the-learning-machine/build.py --full

# The BREAK IT experiment:
python projects/01_the-learning-machine/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 1 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
