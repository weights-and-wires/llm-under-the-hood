# Project 30: Non-Transformer Architectures (Mamba, RWKV)

## Hook

Every model in this book has been a transformer. Every chapter, from **Project 4: Attention From Scratch** through the routed MoE blocks and the long-context tricks, built on the same idea: attention, the operation where every token looks at every other token.

So here is the question that hangs in the air. Are transformers the only way to build a language model? What happens if attention is not the answer?

This chapter takes the assumption apart, replaces attention with something different, trains the result on the same Shakespeare data, and shows exactly what you win and what you lose.

I want to put my opinion in the open before we start. Transformers are not the only path. They are the path that won the 2017 to 2024 round. There is no theorem that says they will win the next one, and a reader who finishes a book on LLM engineering in 2026 without knowing what an SSM looks like under the hood is reading a book that already has a hole in it.

## The Concept

Picture two people reading a long document.

The first person is a transformer. Every time they think about a word, they flip back through the whole document and re-read every page. They can see anything they want, at any distance. When the document is short, this is fine. When the document is a thousand pages, they spend most of their time flipping back through earlier pages instead of reading new ones. The cost grows with the square of the length, because each new word makes them re-read everything that came before.

The second person is taking notes. They have one notebook, a fixed number of pages, and they update it as they read. When they reach a new word, they decide what to keep in the notebook and what to discard. They never flip back through the original document. The notebook is their entire memory of what came before. Each new word costs the same constant amount of work — read the word, update the notebook, move on.

The second person is a **state-space model**. A state-space model, or SSM, is a sequence model where the entire history is compressed into a fixed-size vector called the **state**, and each new input produces a new state by applying a small linear update. The state is the notebook. The update rule is how the notebook gets revised after each word. The reader never goes back.

This buys you something concrete. A transformer processing N tokens does about N² units of attention work. A state-space model processing N tokens does about N units of recurrence work. At sequence length 1,000 the transformer does roughly a million operations of attention; at 100,000 it does ten billion. The state-space model does a thousand and a hundred thousand, respectively. Linear instead of quadratic. This is the whole reason anyone cares.

It also costs you something concrete. The notebook has finite size. If you need to recall an exact six-token quote that appeared at page 1 when you are now on page 1,000, the transformer can flip back and read it; the state-space model can only consult the notebook. If that specific phrase did not get written down, it is gone. **Exact retrieval**, pulling a specific past token back, verbatim, is the thing a fixed-size state cannot guarantee, no matter how cleverly the update rule is designed. State compression is a tradeoff: speed and memory for perfect recall.

This is the part I had to rewrite the most. My first draft of this section sold the speed story too hard and then mumbled about recall as a footnote. Reading my first draft back, I came away thinking SSMs were strictly better, which is exactly the wrong conclusion. So I moved the cost paragraph up to live next to the benefit paragraph. They are not separate stories; they are the same story.

Now refine the picture. The early state-space models had a single, fixed update rule. Every word got summarized into the notebook the same way, regardless of what the word was. That worked badly. A sensible note-taker writes different things depending on what they are reading. A name gets copied down verbatim; a filler word gets ignored. Mamba's main contribution, in a block the authors call **S6**, is to make the update rule depend on the input. The state-update parameters are computed from the current token, so the model can choose what to remember and what to forget on a per-token basis. The technical name for this property is **selectivity**, and it is the thing that lifted state-space models from being curiosities into being competitive language models.

Coming from an RF telecom background (GlowNetworks, 2011 to 2012), the SSM equations were not as alien to me as they might be to a pure NLP reader. A linear state-space update with a learned A matrix is the exact same shape as a first-order infinite impulse response filter. Every telecom system worth shipping has had these in its signal path for fifty years. The clever part is not the recurrence; the clever part is making the filter coefficients depend on the symbol stream itself.

RWKV (pronounced "rwah-kuv") is a different design with the same goal. It blends two ideas: a recurrence that updates a small state, and a channel-mixing block that looks like a standard MLP. The recurrence is called **time-mix** because it mixes information across timesteps, and the MLP is called **channel-mix** because it mixes information across feature channels. Time-mix has an explicit **decay term** — each step, the previous state is multiplied by a per-channel decay factor before the new information is added. There is no selectivity. The update rule is the same for every token. The model is simpler than Mamba, easier to implement, and almost as competitive on language modeling.

A few terms that will come up a lot. A **recurrence** is a computation where the output at step t depends on the output at step t minus one, plus the input at step t. **Discretization** is the trick of taking a continuous-time differential equation and approximating it with discrete time steps. SSMs are written in continuous-time form on paper and then discretized so a GPU can run them. A **state-space model** is any model with the form `state_new = A · state_old + B · input` and `output = C · state_new` for some matrices A, B, C. **Selectivity** means those matrices depend on the input. **Time-mix** mixes across the time axis; **channel-mix** mixes across the channel axis. Everything in this chapter is a variation on those five terms.

One last analogy before the build. A transformer is a person who can re-read the whole document every time they think about a word. A state-space model is a person taking notes that summarize what matters as they go. The transformer never forgets; it just takes a long time to remember. The state-space model never slows down; it just cannot recall what it did not write down.

![Figure 30.1. State update as a notebook. Top: transformer re-reads the document on every token, growing work quadratically. Bottom: SSM reads each token once, updates a fixed-size state, and moves on.](figures/fig_ssm_state_update.png)

## Why It Matters

Three reasons.

The first is cost. A transformer at sequence length 32,000 spends almost all of its time and memory in attention. At sequence length 100,000, attention dominates so heavily that the rest of the model is rounding error. Long-context language modeling, retrieval-augmented systems, audio modeling, DNA modeling, and code modeling all push past the comfort zone of standard attention. **Project 16: Long-Context Extension (RoPE, YaRN, NTK-Aware)** showed how to stretch a transformer to longer contexts; this chapter shows what an architecture designed for long contexts from the start looks like.

The first time I profiled a long-context transformer, attention dominated the budget more brutally than I had bargained for. At 32K tokens roughly 70 percent of forward-pass time was attention; at 100K it was over 90 percent. The rest of the model essentially does not exist at that scale.

The second is shape. The KV cache of **Project 13: Fast Inference: The KV Cache** is a beautiful trick, but its memory grows linearly with the number of tokens generated so far. A long generation eats memory until it dies. An SSM has a fixed-size state. The "cache" never grows. For server workloads that hold many conversations open at once, that is the difference between fitting four sessions on a GPU and fitting forty.

The third is intellectual hygiene. Every successful idea in deep learning eventually gets challenged by a different idea that does the same job a different way. Transformers earned their dominance because the alternatives at the time, vanilla RNNs and LSTMs, could not match them on quality at scale. That was true in 2017. It is less obviously true in 2024. Mamba, RWKV, S4, H3, RetNet, and a small zoo of related architectures all claim transformer-level quality on language modeling at a fraction of the long-sequence cost. Knowing what is in that zoo, what its tradeoffs are, and what specific tasks it loses on is now part of being literate in the field.

This chapter does not promise that transformers are obsolete. They are not. The retrieval test in BREAK IT will show why. But the reader who finishes this chapter knows what a non-transformer language model looks like under the hood, can train one on the same data they used in **Project 5: Your GPT From a Blank File**, and can read the four major papers in the area without flinching.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/30_non-transformer-architectures-mamba-rwkv/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/30_non-transformer-architectures-mamba-rwkv/build.py --full

# The BREAK IT experiment:
python projects/30_non-transformer-architectures-mamba-rwkv/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 30 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
