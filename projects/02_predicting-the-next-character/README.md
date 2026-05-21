# Project 2: Predicting The Next Character

## Hook

Why does a language model that knows nothing about meaning, facts, or the world still manage to produce text that looks eerily language-like? If all it does is guess one next character at a time, why does that not collapse into nonsense immediately? And if that really is all it does, what changes when you move from a dumb counting table to a neural network with learned embeddings? This project answers that by stripping language modeling down to its smallest honest form: given a few characters, predict the next one.

The first time I trained one of these character models, I expected the bigram baseline to be hopeless and the neural model to be obviously better. The opposite happened on my first run. The neural model overfit on a tiny names file, the bigram model held up because it cannot overfit much, and the lesson stuck: baseline models are not toys. They are sanity rails.

## The Concept

Start with the least mystical version of language generation possible.

Imagine a jar full of paper slips. On each slip is a pair of characters you saw in some text: `th`, `he`, `er`, `qu`, `a `, `, `, and so on. If I show you the first character and ask what usually comes next, you can look through the jar, count what followed that character before, and make a guess.

That is a language model in miniature. Not a chatbot, not a reasoning engine, not a world simulator. A next-step guessing machine. At character level, the model does not think in words. It sees pieces like `t`, `h`, `e`, space, newline, punctuation. Its job is brutally narrow: given the previous context, assign probabilities to possible next characters. If the current character is `q`, the next one is often `u`. If the current character is a newline in a names dataset, the next one is often the first letter of a name. That is the whole job.

### The first version: counting

A **bigram** model is the simplest useful version. "Bigram" just means "pair of tokens." Here, the tokens are characters, so a bigram model asks: when I see character `c`, what character tends to come next?

You can picture it as a spreadsheet where rows are current characters, columns are next characters, and cell `(row='q', col='u')` stores how often `u` followed `q` in the training data. To generate text, you look at the current character, read the row, turn counts into probabilities, and sample the next character.

This model has no deeper understanding. It does not know that `the` is a word. It does not know grammar. It just remembers local transitions. And yet it already produces something that feels language-shaped, which matters, because it means a surprising amount of language surface form lives in local statistics alone.

The first time I noticed this I felt vaguely cheated. The model had learned nothing in any reasonable sense. It just kept a count book. And the count book alone produced output that, at a glance, looked like a child practicing a foreign alphabet. That is information about language, sitting in plain sight, that I had been mentally crediting to something more sophisticated for years.

### The second version: learned representations

A counting table treats every character as a separate bucket. `a` is one row, `b` is another, `z` is another. The model gets no opportunity to notice that vowels behave somewhat similarly, or that punctuation marks share some structural role, or that common first letters cluster together. A neural network changes that.

Instead of storing one row per character directly, it gives each character an **embedding**: a list of numbers that represents a token. Think of the embedding as a coordinate in an abstract space, or a learned identity card made of numbers rather than a one-hot "I am character 17" label. Something like:

- `a -> [0.7, -1.2, 0.1, ...]`
- `e -> [0.8, -1.0, 0.2, ...]`
- `q -> [-0.4, 2.1, -0.7, ...]`

The network learns those numbers so that characters which need to be treated similarly can land in similar regions of that space. The model no longer just memorizes counts. It learns a geometry. That is the first real step toward "representation learning."

### Why embeddings matter

Suppose you run a restaurant and every customer is described by a single ID number. Customer 481, customer 938, customer 221. If you want to guess what each person will order, IDs are not very informative unless you memorize every person separately.

Now suppose instead each customer comes with a profile: likes spicy food, prefers vegetarian dishes, visits on weekends, and orders dessert often. Now similar customers can share patterns. Embeddings do this for tokens. The model gets to say, in effect, "these two characters behave alike in some ways, so I can share statistical strength between them." That phrase, sharing statistical strength, is one of the main reasons neural models beat raw counting. A model that can generalize from what it knows about `e` to handle a context involving `i` has learned something the counting table never could.

I think of embeddings as a small notebook the model writes for itself, one row per token, where each row records "things I have noticed about this character." The notebook starts random. Training fills it in. The notebook is the bridge between the dumb integer ID and the rich behavior the network needs the token to have.

### What the model actually predicts

At each step, the model outputs a score for every possible next character. If your vocabulary contains 65 characters, the model produces 65 scores, then turns those scores into probabilities with **softmax**. Softmax converts arbitrary scores into a probability distribution that sums to 1. If the scores strongly favor one character, the output distribution is sharp; if the scores are similar, the distribution spreads out.

Then you either take the highest-probability character, or sample from the distribution. Sampling is what makes generation varied. If you always pick the top answer, the model becomes rigid and repetitive; if you sample too freely, it becomes chaotic. That tradeoff leads us to **temperature**.

### Temperature: how risky the model behaves

Temperature is a knob you apply during sampling to control how boldly the model draws from what it has learned. A plain analogy: imagine a spinner with weighted slices. At normal temperature, the slice sizes match the model's probabilities. Lower the temperature, and the biggest slices get even bigger. Raise it, and the smaller slices grow fatter.

At `temperature = 0.1`, the model almost always picks the safest continuation. At `temperature = 2.0`, even weak options get much more chance. You are not changing what the model learned. You are changing how you sample from it.

This is one of those settings users meet first in a chat UI slider, with no explanation of what it does. The slider is doing one specific thing to the logits before sampling. Once that one fact lands, the slider stops being magic and starts being a knob with a behavior you can predict.

### Negative log-likelihood: how wrong the model is

We need one number that answers: how surprised was the model by the actual next character? That number is the **negative log-likelihood**.

The intuition first. If the true next character is `e` and the model assigned probability `0.9` to `e`, that is a confident good guess. If it assigned probability `0.01`, that is a bad miss. We want a penalty that scales accordingly: high probability on the correct answer means a small penalty; low probability means a large penalty. Negative log does exactly that. If `p` is the probability the model assigned to the correct next character, the loss for that example is:

$$
L = -\log p
$$

Now earn the symbols:

- `L` is the loss: one number measuring how wrong the prediction was
- `p` is the probability the model assigned to the correct next character
- `log` means logarithm

Why log? Because it punishes extreme confidence when wrong much more harshly than mild uncertainty. If the model says "I am nearly certain the next character is `x`" and the truth is `e`, it should pay a big price.

A few concrete values make this click:

- if `p = 1.0`, then `L = 0`
- if `p = 0.5`, then `L ≈ 0.69`
- if `p = 0.1`, then `L ≈ 2.30`
- if `p = 0.01`, then `L ≈ 4.61`

So lower is better. This metric will follow us through the whole book in different forms.

## Why It Matters

Once you build a character model yourself, several foggy ideas become concrete.

First, you stop imagining that text generation begins with huge models and giant datasets. It begins with the humble act of predicting the next token. Everything else is scale, architecture, and training technique layered on top.

Second, you see that "knowing language" has levels. A bigram model knows almost nothing, but it still captures local regularities. A neural model knows more because it can learn useful internal representations. That difference between memorizing observations and learning representations is a major dividing line in machine learning.

Third, you finally have a place to attach the word "embedding." Without this chapter, "embedding" sounds abstract. Here it becomes practical: a learned numeric identity for each character that lets the network treat different inputs differently.

Fourth, temperature stops sounding like a mysterious generation setting in a UI. It becomes a concrete control over randomness in sampling.

The break-it experiment matters even more than the successful build. If you force all characters to share the same embedding, the model loses the ability to tell them apart. That failure teaches you more about embeddings than any definition does. It shows that embeddings are not decoration. They are the mechanism that gives distinct inputs distinct internal identities.

Most beginner explanations of embeddings stop at "a vector that represents meaning." That is fine as a one-liner and useless as a working definition. The thing that gives an embedding its job is the lookup, the fact that distinct IDs route to distinct rows. Break the routing and the rest of the network has nothing to do.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/02_predicting-the-next-character/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/02_predicting-the-next-character/build.py --full

# The BREAK IT experiment:
python projects/02_predicting-the-next-character/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 2 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
