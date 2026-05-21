# Project 14: Speculative Decoding

## Hook

**Project 13: Fast Inference: The KV Cache** made generation fast by remembering everything the model had already attended to. You thought that was the end of the story. You had cached every key, every value, and brought the per-token cost down to a single attention pass.

Then someone tells you their model produces the exact same output yours does, token for token, but two to three times faster. No quantization. No smaller model. No tricks with the sampling temperature. The output distribution is, by construction, identical.

How is that possible? How can you skip most of your model's forward passes and still get the same tokens as if you had run every one of them?

This chapter is about the trick that makes that work, and about the precise rejection rule that protects you from cheating in the process.

## The Concept

Picture a senior engineer reviewing pull requests. The senior is slow, careful, and expensive. A junior engineer drafts the changes (fast, cheap, occasionally wrong). Most of the time the junior's diff is fine and the senior approves it with a glance. Sometimes the junior is wrong and the senior rewrites the section. Either way, the senior reviews many diffs in the time it would have taken to write any one of them from scratch. This is the entire shape of speculative decoding.

The senior is your main model, say a 7-billion parameter transformer. The junior is a **draft model**, which is just a smaller, cheaper version of the same kind of model, usually trained on the same data with maybe one tenth the parameters. The draft model proposes a short run of tokens; the main model reviews them all at once. When the main model agrees, those tokens are free. Well, almost free. They cost the main model exactly one parallel forward pass, regardless of how many tokens the draft proposed. When the main model disagrees, you fall back to a single token from the main model, throw away the rest of the draft, and start over.

The first time I saw this scheme described, my honest reaction was suspicion. It felt like a sleight of hand. Two models giving you the answer of one, faster, for free. Once I worked through the rejection-rule proof in Step 4 it went from suspicious to inevitable, but I want to flag that confusion as legitimate: this trick is harder to believe than it is to implement.

Two pieces of jargon to settle now. The number of tokens the draft proposes in each round is **K**, usually somewhere between 4 and 8. The fraction of proposed tokens the main model accepts on average is the **acceptance rate**, written as α. A run with K=5 and α=0.7 means that on average 3.5 of the 5 draft tokens get accepted per round, which means the main model produces roughly 4.5 tokens per forward pass on average (3.5 accepted draft tokens plus 1 bonus token the main model emits after rejection). Compare that to standard autoregressive decoding, which produces exactly one token per forward pass, and you can see where the speedup comes from.

The cheap part is intuitive. The careful part, the part this chapter exists to teach, is the rejection rule. The naive instinct is to say: if the draft picked token *t* and the main model would have also picked *t*, accept it. That sounds right and it is dead wrong. The main model is not picking one token. It is producing a distribution over the entire vocabulary at every position, and a sampler turns that distribution into a single token. To say "the main model picked *t*" you would have had to roll the dice, and rolling the dice already commits you to one of many possible answers. To preserve the main model's distribution exactly, you cannot just check what the main model picked. You have to check the probability the main model would have assigned to *t* and apply a precise rule that accepts or rejects accordingly.

Here is the rule, stated cleanly and then unpacked. For each draft token *t* at position *i*, let *p* be the main model's probability of *t* at that position and *q* be the draft model's probability of *t* at that position. Sample a uniform random number *u* between 0 and 1. Accept *t* if *u* < min(1, *p*/*q*). If you reject, do not just take the main model's most likely token; instead, sample from a particular **residual distribution** that we will define carefully below.

Why this rule? Walk through it for a moment. If *p* is greater than or equal to *q*, the main model likes the token at least as much as the draft did, and you accept with probability 1. That is the easy case. If *p* is less than *q*, the draft was over-eager about *t*, and you accept with probability *p*/*q*. That ratio is exactly the correction needed to bring the joint distribution of "what the draft proposed and you accepted" back into line with what the main model would have done on its own. The math here is short and worth seeing later. For now, the intuition: the rule throws away exactly enough of the over-eager draft picks to make the surviving picks look like they were sampled from the main model directly.

The residual distribution on rejection is the second half of the magic. When you reject a token, you have leaked information. The main model has revealed that it disagreed with the draft on this particular token. To preserve the main distribution, you cannot just sample fresh from the main model. You have to sample from a distribution that compensates for what you have already learned. Specifically, you sample from a distribution proportional to max(0, *p*(x) − *q*(x)) over the vocabulary. This is the part of the main model's probability mass that the draft did not already cover. We will work through this carefully in **Step 4**.

The diagram in your head should be something like this. The draft runs forward K times, producing K candidate tokens autoregressively. Then the main model runs forward exactly once, with all K candidate tokens packed into its input sequence. Attention is parallel along the sequence dimension, so the main model produces a full output distribution at each of the K positions in that single forward pass, at the cost of one forward, not K. Then you walk left-to-right through the K positions and apply the rejection rule. The first position where you reject is where this round ends; everything to the right of that position is discarded. After rejection at position *i*, you emit one token from the residual distribution at position *i*. You also get to emit the token from the main model's distribution at position K+1 if all K draft tokens were accepted. That is the bonus token that gives you the "K+1 tokens per forward" payoff in the best case.

![Figure 14.1. Draft-then-verify timing. The draft model runs K cheap forward passes to produce K candidate tokens, then the main model runs one expensive forward pass over all K candidates in parallel. The total wall-clock is K times draft-latency plus one main-latency, compared to K main-latencies for naive decoding. The savings depend on the draft being much cheaper than the main and on most candidates surviving verification.](figures/fig_spec_decode_timing.png)

A couple of named variants are worth flagging up front so you recognize them later. The version above, with a separate small model as the drafter, is the **Leviathan-Chen** style of speculative decoding from the two 2023 papers that introduced it. A later variant, called **Medusa**, drops the separate draft model entirely. Instead, it bolts a small number of extra heads onto the main model itself, trained to predict positions K+1, K+2, K+3 in parallel. The drafter is now built into the main model. You get a similar speedup without the cost of training and serving a second network. We will look at Medusa in **Step 7**.

If you have spent any time around multi-model routing systems, this story will feel familiar. In past multi-model routing work spanning frontier API models, the practical question was always the same: when does a cheap draft answer beat a careful expensive answer, and how do you decide without running both? Speculative decoding is a tighter version of that exact question, with provably no quality loss. The routing work did not give us the same guarantee, which is part of why this trick excites me more than it should.

## Why It Matters

Without speculative decoding, every output token of a large language model costs one forward pass through the entire stack. That cost is the dominant term in inference latency at production scale, and it is what determines how many tokens per second any given GPU can produce. **Project 13: Fast Inference: The KV Cache** brought the per-step cost down by reusing intermediate state, but it could not get you below one forward pass per token. Speculative decoding breaks that floor. With a well-matched draft model, you get two to three output tokens per main-model forward pass on average. The same hardware now serves two to three times the throughput.

A second reason this trick matters is that it costs you nothing in output quality. The acceptance and resample rules are mathematically constructed so that the distribution of generated tokens is identical to what the main model would have produced if you had run it the slow way. This is not "almost the same" or "close enough for most purposes." The proof is short and exact, and we walk through it in Step 4. If your main model produces a particular probability distribution at each position, the speculative procedure produces samples from that exact distribution. No drift. No subtle degradation. The user gets the main model's tokens, just faster.

I want to underline this because most engineering tradeoffs do not look like this. Usually faster means worse, in some dimension you have to defend in a design review. Speculative decoding really does give you a free lunch in the output-quality dimension. The catch is somewhere else, and the BREAK IT section is where it lives.

The trick is everywhere in production serving stacks now. Most large model APIs you call through OpenAI, Anthropic, or open-source servers like vLLM use some flavor of it. **Project 15: Grouped Query Attention** is about making the per-forward cost smaller; speculative decoding is about needing fewer forwards. The two stack cleanly. **Project 17: Production Serving: Continuous Batching, PagedAttention** wires them together with batched scheduling so that a single GPU can serve many concurrent conversations, each using speculative decoding, with KV caches paged in and out of memory as needed. You cannot understand modern inference serving without this chapter.

There is also a clean intellectual reason to learn it. Speculative decoding is one of the cleanest examples in deep learning of a free-lunch technique that works only when a non-obvious condition holds. The condition is that the draft model has to agree with the main model often enough that the work of running the draft is paid back by the work saved on the main. When agreement collapses (when the draft is poorly matched to the main) the trick becomes worse than useless. The BREAK IT section forces you to see exactly when and how the free lunch turns into a tax.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/14_speculative-decoding/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/14_speculative-decoding/build.py --full

# The BREAK IT experiment:
python projects/14_speculative-decoding/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 14 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
