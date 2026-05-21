# Project 3: Building A Tokenizer

## Hook

Why does every serious language model bother with a tokenizer at all? Why not just feed in characters, one by one, and let the model figure it out? Or go the other direction and use whole words, since that is what humans think in? This is where a lot of newcomers get stuck. Characters feel too small. Words feel too rigid. And "**token**" sounds like one of those terms everyone uses before they have explained what problem it solves. This chapter fixes that. You are going to build the thing that decides what the model even sees.

The thing I wish I had known three years earlier: tokenizers are not a side topic. They quietly shape what your model can learn, how much of your context window survives a long document, and how badly your model embarrasses you on languages it was not trained on. When I started building a Tamil triage pipeline for an offline Android deployment, the tokenizer was where most of the early pain lived.

## The Concept

Start with the wrong mental model: "text is made of words."

It is not, at least not in a way computers can trust. Take these three strings:

- `play`
- `playing`
- `replaying`

A whole-word system treats these as three unrelated items unless all three appear in its vocabulary. Now take `color` and `colour`. Same meaning, different spelling. Now take `printf(`, `user_id`, and `2026-03-26`. These are not normal "words" at all, but they appear constantly in code, logs, configs, docs, SQL, shell commands, and chat transcripts. So whole words break the moment text gets messy.

Characters look safer. Every string is made of characters, with no out-of-vocabulary problem and no need to guess word boundaries. But character-level text is like shipping a sofa one screw at a time. A character tokenizer turns `internationalization` into a long chain of tiny pieces. The model must spend compute predicting letter after letter after letter, even when the larger chunk is common and stable. That wastes context window and wastes learning capacity. So we need a middle ground.

Here is the plain-English picture. Imagine you are compressing sticky notes from a busy office. At first, every note is stored as individual letters. That works, but it is wasteful. You keep seeing the same letter pairs: `th`, `he`, `in`, `er`, `re`. So you invent shorthand cards for those pairs. Then you notice bigger chunks appear constantly: `the`, `ing`, `ion`, `printf`, `://`, `.com`. You add cards for those too. You never hard-code English grammar. You never tell the system what a noun is. You just keep asking one question: "What adjacent pair shows up most often? If I replace that pair with a single new symbol, do I save space?"

That is **byte-pair encoding**, or BPE.

BPE starts from raw bytes. A byte is just a number from 0 to 255. Text files on disk are bytes. UTF-8 characters are bytes. Emojis are bytes. Code is bytes. Weird punctuation is bytes. Starting from bytes means nothing is out of vocabulary. Every possible input can be represented, because every possible input is bytes. Then BPE learns a vocabulary by repeated merging:

1. Count all adjacent byte pairs in the training corpus.
2. Find the most frequent pair.
3. Replace that pair everywhere with a new token ID.
4. Repeat until you reach the target vocabulary size.

If you start with 256 byte values and add 768 merges, you get a vocabulary of 1024 tokens. The key idea is simple: common chunks become single tokens; rare chunks stay split. A good tokenizer does not try to make every weird string into one giant token. It gives short, common patterns their own entries and leaves rare junk as smaller pieces. That is why BPE handles both everyday language and ugly real-world text.

Here is the other picture that helps. Think of vocabulary size as drawer size in a workshop. Too few drawers: every tool is broken into tiny parts and scattered everywhere, so you spend forever assembling what you need. Too many drawers: you create a custom drawer for every slightly different object, and most drawers hold one thing and never get opened again. A tokenizer needs the middle ground. Enough drawers to store recurring chunks, not so many that the system memorizes trivia.

Before any code, define the two things you are optimizing against each other:

- Sequence length: how many tokens it takes to represent text.
- Token frequency: how often each token appears in the corpus.

A larger vocabulary usually shortens sequences, but it also creates rarer tokens. That tradeoff is the whole chapter, and Figure 3.1 shows how BPE navigates it by starting from bytes and merging upward.

![Figure 3.1. Byte-pair encoding starts from bytes, merges frequent adjacent pieces, and tries to shorten sequences without wasting vocabulary on one-off chunks.](figures/fig_tokenizer_bpe_flow.png)

## Why It Matters

The tokenizer is not a preprocessing footnote. It changes what the model can learn efficiently.

If your tokens are too small, the model wastes attention on low-level assembly. Suppose the sentence is `The database connection timed out again.` A bad tiny-token setup might turn this into dozens of pieces. The model now spends context and compute reassembling `connection` from fragments instead of learning the larger pattern that `database connection timed out` often appears together.

If your tokens are too large, you get the opposite failure. Imagine a vocabulary that contains bizarre one-off chunks like `timed out again.` or `base connection timed out ag`. Those might save a token or two in the training corpus, but each appears so rarely that the model cannot learn a good embedding for them. An embedding is a list of numbers that represents meaning. If a token appears twice in the whole corpus, the model gets almost no chance to learn what that list of numbers should be.

This was the failure mode that ambushed me during early per-language tokenizer choices in some cooperative fine-tuning experiments. We had merged a vocabulary that looked great on training compression and quietly contained thousands of tokens that appeared two or three times. The training loss told one story. Generalization told another. The embeddings for those rare tokens were essentially noise.

This affects everything downstream. Bad tokenization means longer sequences, more memory cost, less useful context window, harder prediction targets, noisier embeddings, and worse compression of repeated patterns.

There is also a practical reason to care. When people say "this model has a 128K context window," that is 128K tokens, not 128K words and not 128K characters. If your tokenizer is inefficient, the same document consumes more of that budget. A code-heavy file might fit within one tokenizer and overflow in another. So vocabulary design is not cosmetic. It changes how much text the model can see, how often it reuses learned chunks, and how much training signal each embedding receives.

Strong opinion: most "this model is better at X" benchmarks would tell a different story if both models were retokenized on the same vocabulary. Tokenizer-driven differences masquerade as model-driven differences all the time. After enough of these experiments I now find myself instinctively asking "whose tokenizer?" before "whose model?" when reading a comparison.

You can think of a tokenizer as the interface between raw text and learned representation. Project 2 ended with characters as the unit. This project is where "input units" stop being given by nature and become a design choice. That is a major shift.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/03_building-a-tokenizer/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/03_building-a-tokenizer/build.py --full

# The BREAK IT experiment:
python projects/03_building-a-tokenizer/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 3 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
