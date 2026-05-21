# Project 13: Fast Inference: The KV Cache

## Hook

Why does a language model get slower and slower the longer it talks, even though it is only producing one new token at a time? If the model already read the first 500 tokens, why does token 501 seem to make it reread all 500 again? That sounds stupid because it is stupid.

Left alone, autoregressive generation really does keep recomputing work it already did. The KV cache exists to stop that waste. Without it, "generate one more token" secretly means "re-run most of the sequence again." With it, generation stops acting like someone rereading the whole chat history before every reply and starts acting like someone who kept notes.

I spent a chunk of 2025 writing KV-cache-adjacent code on Apple Neural Engine, where we had to swap LoRA adapters in through IOSurface buffers between decode steps. The thing that struck me, and that I want this chapter to land for you, is how mundane the cache turns out to be once you see it. It is bookkeeping. It is one of the most consequential optimizations in modern inference, and it is bookkeeping.

## The Concept

Start with the wrong mental model, because a lot of people have it.

Attention is not memory in the durable sense. It does not keep a little brain-state from one decoding step to the next. It recomputes relationships between tokens inside the current input. If you call the model again on a longer sequence, it will happily recalculate the same keys and values for the old tokens unless you explicitly save them. That saved state is the **KV cache**. The name is dry. The idea is not.

Think of a courtroom transcriptionist. Every time a new sentence is spoken, the judge does not ask the transcriptionist to retype the entire hearing from the beginning. The old transcript is already there. The new sentence gets appended. When someone needs context, they look back at the transcript. That is KV caching.

The model has already turned past tokens into two useful things:

- **keys**: what past tokens contain (the same key projections computed during prefill that are now being cached).
- **values**: what past tokens contribute when selected.

For a new token, the model mainly needs a fresh **query**: what the new token is looking for, the only fresh attention projection during decode. Then it can compare that new query against all the saved keys, pull information from the saved values, and move on. No need to rebuild the old keys and old values every single step.

Let's define the pieces in plain English before any code.

A **token** is the basic unit the model works with. It might be a word, part of a word, punctuation, or whitespace. A **hidden state** is the model's internal representation of a token at some layer — a list of numbers that tries to capture what that token means in context. Query, key, and value are three different linear projections of that hidden state — introduced in **Project 4: Attention From Scratch** — giving the same token three different "views" of itself.

In self-attention, every token asks: "Which earlier tokens matter for understanding me right now?" The token does that by comparing its query to every earlier token's key. The better the match, the more it reads from that token's value. Here is the part people miss. For old tokens, those keys and values do not change during inference. If token 17 already produced a key and a value at layer 6, then tokens 18, 19, and 400 can keep reusing them. Recomputing them is wasted work. That is why the cache stores K and V, not Q. The new token's query changes every step because the new token is new, but the older tokens' keys and values are stable for the current prompt.

The first time I really sat with that idea, I had a small moment of "wait, that's it?" Yes. That's it. The most consequential inference optimization in modern LLMs is "do not recompute the things that did not change." Most beginner explanations skip past this with the air of someone explaining a magic trick, when the trick is just notetaking.

So the KV cache is a per-layer notepad. Each layer keeps a growing stack of old keys and old values. When a new token arrives, the model computes its hidden state through the network, computes its new query, key, and value in each attention layer, appends the new key and value to that layer's cache, and runs attention using the new query against all cached keys and values. That is the whole trick.

The result feels almost unfair once you see it clearly. Without caching, generating token 501 means recomputing attention inputs for tokens 1 through 500 again, rebuilding their keys, rebuilding their values, and only then producing token 501. With caching, it means computing the new token once, comparing it to the stored past, and appending one more row to the cache. Same output goal. Very different amount of work.

Most explanations frame the speedup as a compute win. It mostly is not. Once the cache exists, the real story turns into memory bandwidth: how fast you can read the cached K and V out of HBM at every decode step. That bandwidth bottleneck carries the back half of this chapter.

Now earn a little math. Suppose your prompt length is `T`. In naive generation, step 1 runs on length 1, step 2 on length 2, step 3 on length 3, and so on until length `T`. That total work looks like `1 + 2 + 3 + ... + T`, which grows roughly like `T^2`. So generation time grows much faster than sequence length. With a KV cache, each new step mainly processes one new position against stored history. You still pay some attention cost against the history, but you stop recomputing all earlier K and V projections from scratch. In practice, that is the difference between "why is this chatbot stalling?" and "this feels instant enough to use."

There is another reason the cache matters: it makes the real bottleneck visible. Once you stop wasting compute, memory becomes the wall. The cache grows with context length — every new token adds more stored keys and values for every layer. So the question changes from "can the model in principle look back 32K tokens?" to "can I afford to store 32K tokens worth of keys and values for every layer, head, and batch item?" That is why context length is not just an architectural idea. It is a memory budget. Figure 9.1 makes the contrast between the naive and cached approaches concrete.

![Figure 9.1. The KV cache removes repeated recomputation by storing old keys and values and appending only the new token state at each decode step.](figures/fig_kv_cache_compare.png)

## Why It Matters

Without a KV cache, long-context generation is slow enough to feel broken. You can prove this to yourself with a toy example. Ask a tiny model to generate 256 tokens from a 512-token prompt. Naive decoding keeps rerunning almost the entire prefix, so the later steps cost much more than the early ones and throughput falls off as the sequence gets longer. The user does not care why. The user sees lag. That alone would be enough reason to care.

But the cache matters for more than speed. If you run a model for chat, coding, or agents, you are not doing one forward pass over a fixed sequence. You are doing incremental decoding over and over. That means your runtime lives or dies by how cheaply it can carry the past forward, and the KV cache is the single most important inference optimization because it turns repeated work into stored state.

I spent two years in healthcare release management where the operational rule was "zero downtime, no deploy windows." That experience taught me something that transfers awkwardly well: production inference cares about p99 latency, not throughput averaged over a benchmark. A cache that drops a tail of slow requests by 50% is worth more than one that lifts the mean by 10%. Keep that in mind as you measure later in this chapter.

The cache also exposes a real memory trade that people often miss. A model might fit in VRAM and still fail under long contexts because the KV cache grows until memory runs out. The model weights are one cost; the cache is another, and they are independent. For one layer, the cache stores keys with shape roughly `[B, H, T, d_head]` and values with shape roughly `[B, H, T, d_head]`, where `B` is batch size, `H` is number of attention heads, `T` is cached sequence length, and `d_head` is the size of one head. That means cache size grows linearly with `T`. Double the context, roughly double the cache memory. Double the batch size, double it again. Double the number of layers, double it again because each layer has its own cache. This is why the words **context window** often hide a hardware statement. A model can only keep as much usable past as memory allows.

Then there is the sliding-window trick. If you drop the oldest cache entries and keep only the most recent `W` tokens, memory stops growing after `W`. You traded full long-range context for bounded memory. Sometimes that is a good trade. Sometimes it destroys quality. For local text continuation, a 2K or 4K sliding window may be fine. For code generation that depends on an import 8K tokens earlier, it may be terrible. The task decides.

Finally, the cache matters because if it becomes corrupted, generation collapses in a way that feels spooky until you understand why. One bad cached key in the middle of the sequence can poison every later attention lookup that touches it. You are not damaging one token. You are damaging a persistent reference point that future tokens may keep consulting. A bad cache does not heal itself just because later math is correct. It keeps feeding wrong context into every later step.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/13_fast-inference-the-kv-cache/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/13_fast-inference-the-kv-cache/build.py --full

# The BREAK IT experiment:
python projects/13_fast-inference-the-kv-cache/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 13 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
