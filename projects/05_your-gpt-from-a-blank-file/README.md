# Project 5: Your GPT From A Blank File

## Hook

Why does a GPT feel mysterious right up until the moment you try to write one yourself? You already built the pieces: gradients, a tiny language model, a tokenizer, and attention. So why does opening a blank file still feel like stepping off a curb you cannot see?

Because the hard part is no longer any one component. The hard part is the glue. A GPT is not "attention plus some layers." It is a whole system where token embeddings, positional information, residual paths, normalization, initialization, batching, optimization, clipping, checkpointing, and generation all have to cooperate. Miss one line and the whole thing does not become slightly worse. It becomes nonsense, NaNs, or wasted hours.

I wrote my first GPT three weekends in a row before I had something I trusted. The first weekend I shipped a file that looked correct, trained, produced text, and was quietly leaking validation data into the training stream because I had reused one index variable in two places. Until you have written this yourself once, you do not know how many of those traps the polished code is hiding from you.

## The Concept

Here is the plain-English picture of what we are assembling.

A tokenizer turns raw text into tokens, which are just the units the model sees. Maybe they are characters. Maybe they are subword pieces like `ing` or `tion`. The model never sees "language" directly. It sees token IDs, and everything else is built on top of that integer sequence.

An embedding table is a lookup table that turns each token ID into a list of numbers. That list is the token's current internal representation. If token ID `42` means `"king"`, the embedding table maps `42` to some vector in `d_model` dimensions. If `d_model = 256`, then `"king"` becomes 256 learned numbers that will shift throughout training as the model figures out what "king" means in context.

Position matters too. `"dog bites man"` and `"man bites dog"` use exactly the same tokens in a different order. So the model adds position information alongside token identity: if token embeddings say *what* a token is, **positional embeddings** say *where* it sits in the sequence.

Then the **transformer blocks** begin. Each block does two jobs: first, attention lets each token decide which earlier tokens matter right now; second, the feedforward network does local thinking on each position after attention has already mixed in context from elsewhere. These two sublayers are wrapped in residual connections, the `x + F(x)` skip lane that carries the old signal forward by simple addition. If the new sublayer is not helping yet, the old information still passes through untouched. LayerNorm keeps the scale of activations under control at each step, functioning like electrical regulation. Without it, one layer can push values into a range the next layer cannot sensibly consume. Figure 5.1 shows how attention and the feedforward network both feed back into the residual stream.

![Figure 5.1. A transformer block is a residual stream with two corrections: attention mixes information across positions, and the feedforward network transforms each position locally before both updates rejoin the running hidden state.](figures/fig_transformer_block_residual_stream.png)

After N of these blocks, the model has a final hidden state for every position in the input sequence. Then it projects that hidden state into vocabulary-sized logits: one score per possible next token. The score for `"e"` might be high, the score for `"%"` might be low. Softmax turns those scores into probabilities.

Training is next-token prediction. If the input is:

```text
To be, or not to b
```

The target is:

```text
o be, or not to be
```

The model predicts the next token at every position, compares those predictions to the true next tokens, and computes a loss: one number that tells you how wrong it is. Then backpropagation assigns blame through the whole network, the optimizer nudges the parameters, and you do it again, ten thousand times or more. That cycle, repeated continuously, is GPT training in one breath: guess the next token, measure the mistake, assign blame, update the numbers.

The first time I watched this loop run on my own code, I expected the loss to fall in a clean diagonal. It does not. It bounces. Sometimes it bounces for two hundred steps before it commits to a direction. That bouncing is a feature, not a bug, but you do not learn that from a textbook curve.

So why does this chapter matter as a blank-file exercise? Because until you write the whole thing yourself, "GPT" is still a bag of named parts in your head. After you write it, you know where the batch comes from, why the targets are shifted by exactly one position, where **weight tying** saves parameters, why learning rate schedules are not decoration, why clipping exists, why initialization is part of the model itself, and why checkpointing is not optional if your time matters. You stop thinking "a GPT has transformer blocks" and start thinking "a GPT is a training system that predicts one token at a time under very specific constraints." That is a better mental model to carry into everything that follows.

## Why It Matters

Without building this yourself, the next reference implementation will lie to you by accident. Not because the code is wrong, but because production code hides pain. A good reference implementation makes dozens of choices look inevitable. If you meet those choices only inside polished code, they look like style. They are not style. They are survival.

This is the strongest opinion I will commit to in the whole book: most "from scratch" tutorials I have read fail right here. They give you the transformer block and skip the wiring around it, and the wiring is where every painful weekend of my life has been spent.

This project matters because it turns the transformer from architecture into system. When training loss drops, you know that is not one knob. It is the result of many agreements being kept at once: the tokenizer gives sensible units, the batch sampler does not feed garbage windows, initialization does not trap the model in symmetry, the learning rate is high enough to learn but not high enough to explode, gradient clipping catches rare spikes before they wreck the run.

If you can write a GPT from a blank file, you stop treating language models as sealed machines. You can now ask: Why did validation loss stop improving? Is this instability optimizer state or model code? Did I really resume training, or only reload weights?

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/05_your-gpt-from-a-blank-file/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/05_your-gpt-from-a-blank-file/build.py --full

# The BREAK IT experiment:
python projects/05_your-gpt-from-a-blank-file/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 5 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
