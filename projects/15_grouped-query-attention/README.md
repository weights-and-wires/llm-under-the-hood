# Project 15: Grouped Query Attention

## Hook

Why do modern LLMs keep getting cheaper to run without suddenly becoming stupid? If attention is the heart of the model, why do many production models take part of that mechanism, share it across heads, and barely lose anything? That sounds wrong the first time you hear it. If eight heads are good, shouldn't eight fully independent sets of keys and values be better than two shared ones?

I remember the exact moment this clicked for me. I had been reading the LLaMA-2 config and saw `num_key_value_heads` sitting next to `num_attention_heads` with a smaller number, and my first reaction was to assume it was a typo in the model card. It was not. Project 15 is where you learn that a lot of transformer design is not about asking "what is most expressive in theory?" but "what is expensive, what is redundant, and what can we share without wrecking the model?"

---

## The Concept

Start with the ordinary picture of multi-head attention.

You already know that attention lets each token look back at earlier tokens and decide what matters. In multi-head attention, the model does this several times in parallel. One head might focus on short-range syntax. Another might track names. Another might lock onto indentation or punctuation. You can think of heads as several readers annotating the same paragraph for different reasons.

In standard multi-head attention, every head gets its own three tools: a query projection, a key projection, and a value projection (introduced in **Project 4: Attention From Scratch**). What each head looks for, what it says it contains, and what it contributes if selected. That means if you have 8 heads, you have 8 separate query views, 8 separate key views, and 8 separate value views.

Now here is the uncomfortable question: do we really need 8 separate copies of keys and values?

Imagine a meeting where 8 analysts read the same document. Each analyst has a different question in mind. That is the query. One analyst asks "Where is the subject of this sentence?" Another asks "Which earlier token sets the tense?" Another asks "Is this part of a code block?" Those questions really do need to differ.

But now imagine the document itself comes with sticky notes attached to each paragraph: what information this paragraph offers if someone asks about it, and what label helps others find it. Those are the value and key. Do you need 8 totally separate sets of sticky notes on the same document, one for each analyst? Sometimes yes. Very often no. That is the opening GQA walks through.

I cut two earlier versions of this analogy. The first used librarians and was too cute. The second used database indexes and was too narrow. The meeting-room version stuck because it produces one specific click in the reader: ok so the questions stay but the notes get shared. That is the right click. The first two analogies did not produce it.

**Grouped Query Attention** says: keep many query heads, because different heads really do ask different questions — but reduce the number of key/value heads, because many heads can share the same description of the past. So instead of 8 query heads, 8 key heads, and 8 value heads, you might have 8 query heads, 2 key heads, and 2 value heads, with each group of 4 query heads sharing the same K and V. This is why it is called grouped query attention: queries stay more diverse, keys and values get shared.

That sharing matters because during inference, the model stores old keys and values in the **KV cache**, the runtime notepad from Project 13. Once a token has been processed, you keep its keys and values around so you do not recompute them for every next token. If you cut the number of K/V heads, you cut KV cache size.

That is the whole trick. And that trick buys real memory, real speed, and real context length.

Let's make the progression concrete. In standard multi-head attention, you might have 8 query heads, 8 key heads, and 8 value heads, so every head owns its own full K/V stream. In grouped query attention, you might keep the 8 query heads but reduce keys and values to 2 heads each, so four query heads share one K/V pair. In multi-query attention, you push the same idea to the limit: 8 query heads, but only 1 key head and 1 value head shared by all of them. That last case saves the most memory and usually costs more quality than GQA, but often less than newcomers expect.

The first confusion most people have is: "If the keys and values are shared, don't all the heads become the same?" No. The heads still have different queries, which means they still score the same stored information differently. Imagine several people searching the same library catalog. The catalog is shared, but different questions still pull out different books.

A natural worry is that sharing K/V will drag heads toward each other in attention weight space. It does not. When I dumped per-head attention maps on a small GQA run during the writing of this chapter, the heads within a group still produced visibly different patterns. The queries are doing real work; the keys do not need to be unique for the heads to disagree.

The second confusion is: "If shared K/V works so well, why didn't everyone do this from day one?" Because standard multi-head attention is the clean, symmetric version. It is easy to explain, it gives each head full freedom, and early work was still finding out what mattered.

Once people started paying the bill for running large models at scale, attention stopped being only a modeling choice and became a systems choice. That is where GQA wins. My own cooperative-fusion runs across a long set of small-to-mid-size LoRA experiments made this brutally clear before I ever read the GQA paper. The KV cache was the first thing I went after when I needed to fit a longer evaluation context on the same GPU, and I went after it before I touched anything else.

Now we can earn a little notation.

Let:

- `d_model` be the hidden width of the model
- `H` be the number of query heads
- `H_kv` be the number of key/value heads
- `T` be sequence length
- `B` be batch size
- `d_head = d_model / H`

In ordinary multi-head attention, `H_kv = H`. In grouped query attention, `H_kv < H`. The number of query heads stays `H`, but keys and values only need storage for `H_kv`. That means KV cache memory scales with `H_kv`, not `H`. If you change from 8 KV heads to 2 KV heads, you cut KV cache memory by 4x.

That is the first equation worth writing because now it means something:

`KV cache size ∝ B × T × H_kv × d_head`

Read it in plain English: bigger batch size means more cache, longer context means more cache, more KV heads means more cache, bigger head width means more cache. If `H_kv` drops, cache drops linearly. Figure 10.1 shows how grouped query attention achieves that reduction by sharing KV heads across query groups.

![Figure 10.1. Grouped query attention keeps multiple query heads but shares fewer key/value heads, which preserves query diversity while shrinking the KV cache.](figures/fig_gqa_shared_kv.png)

---

## Why It Matters

The KV cache grows with context length because the model keeps old keys and values around for every previous token, and standard attention stores those keys and values separately for every head. That works, but it means the cache expands faster than you want, and the consequences are practical — not theoretical.

First, it eats memory directly. If your model uses 8 KV heads and you drop to 2, you reduce the KV cache by 4x. That can be the difference between fitting a 16K context on one GPU or not fitting at all.

Second, it hurts throughput. Memory traffic is often the enemy during inference. Even when the math itself is cheap, reading and writing all that cached data costs time. A smaller KV cache often means better tokens per second. Both effects compound as context length grows, which is why GQA adoption followed the industry's move toward longer contexts rather than preceding it.

Third, and this is the one that matters for the rest of the book, GQA changes architecture compatibility in a hard way. You can only swap model parts around if their internal interfaces agree, and a GQA model with 8 query heads and 2 KV heads does not have the same weight shapes as an MHA model with 8 query heads and 8 KV heads. You cannot pretend they are interchangeable. The checkpoint will tell you that immediately, with a dimension mismatch. That failure is not a nuisance. It is the lesson.

Architecture is a contract. If one model stores K/V projections at one shape and another expects a different shape, "just load the weights" is not a plan.

GQA also illustrates a broader rule that keeps appearing in LLM engineering: symmetry is nice for whiteboards, but production systems reward asymmetry that saves bandwidth. Queries need to stay flexible. Keys and values can often be shared. Modern LLMs use that trade because it works.

My strong opinion on this, after watching engineering teams reach for it again and again: GQA is one of those rare changes where the production benefit is enormous and the modeling penalty is so small that pretending it is a "tradeoff" almost overstates the cost. The honest framing is closer to "we found free money under the couch."

---

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/15_grouped-query-attention/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/15_grouped-query-attention/build.py --full

# The BREAK IT experiment:
python projects/15_grouped-query-attention/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 15 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
