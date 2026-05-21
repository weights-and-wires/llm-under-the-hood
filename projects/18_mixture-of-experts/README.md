# Project 18: Mixture Of Experts

## Hook

Why would anyone build a model with parts that stay asleep most of the time? That sounds wrong on purpose. You spend money training millions or billions of parameters, then for each token you wake up only a small fraction of them. If dense models already work, why add a router, a pile of experts, new failure modes, and a fresh way for training to go off the rails?

The short answer is that one giant feedforward network makes every token pay for every bit of knowledge, even when most of that knowledge is irrelevant. Mixture of Experts fixes that by turning one big general-purpose worker into a team of specialists, then forcing a router to decide who should handle each token.

I spent the better part of last year on this problem. A preprint I wrote on evolutionary Mixture-of-LoRA was, essentially, a long argument about which part of an MoE-LoRA system was actually doing the work. The honest answer surprised me. I will get to that.

## The Concept

Start with a normal transformer block. After attention, each token goes through a feedforward network, often called the FFN. You can think of the FFN as a workshop that every token visits alone. Attention decides what context matters, and the FFN turns that context into something useful. It is where a lot of the model's factual and pattern-matching behavior lives.

In a dense model, every token enters the same workshop. That means the token `"def"`, the token `"mitochondria"`, and the token `"therefore"` all use the same set of weights in the FFN. The workshop has to be decent at everything because it serves everyone.

Mixture of Experts changes the workshop into a building with multiple rooms, each with its own specialist team. One room might get better at code-like patterns. Another might drift toward names and entities. Another becomes good at punctuation-heavy structure. Nobody assigns these roles by hand. The model discovers them during training, sometimes in ways that look nothing like the clean taxonomy you imagined. The important design change is simple: each token does not go through all rooms. A router looks at the token's hidden state, scores the experts, picks the top few, and sends the token only there.

I originally assumed expert specialization would look like recognizable domains. Code expert here, prose expert there. After staring at routing histograms for a few weeks during that work, I gave that up. The categories the router invents are weirder, narrower, and sometimes embarrassingly token-level. One expert in one of our runs became, basically, the "open-paren" expert. That is not a joke.

So instead of one feedforward network for every token, you get many feedforward networks total, but only a few active per token. That is sparse computation — most of the parameters exist, but only a subset participates in any one forward pass.

If dense FFN is one huge call center where every question hits the same staff, MoE is a dispatch system with specialists. The dispatcher listens to the question and sends it to the top one or two departments. The company can employ far more expertise overall without putting the full staff on every call. That is the whole economic argument for MoE: more total parameters, but roughly similar active compute per token.

Now the confusion that usually shows up. If only some experts handle each token, doesn't that make the model fragile? And doesn't the router just learn to send everything to one expert? Yes, on both counts. If one expert is great, why not let it do all the work? Because then the other experts stop learning, and your fancy specialist building turns back into one crowded room with extra rent.

That failure has a name: router collapse. The router starts favoring one or two experts for most tokens. Those experts get more gradient updates, so they improve faster. Because they improve faster, the router prefers them even more. Positive feedback kicks in. The rich get richer. The neglected experts become dead weight.

This is why MoE is not "add more FFNs." The router is now part of the learning problem. I will say it more strongly: most beginner explanations of MoE are honestly terrible because they describe the experts and treat the router as a footnote. The router is the project. Get the router wrong and the experts may as well not exist.

### What the Router Actually Computes

You already know the hidden state of a token is a list of numbers representing what the model currently "thinks" about that token in context. Call that hidden state `x`. The router is just a linear layer that takes `x` and produces one score per expert. If you have 4 experts, the router turns one token representation into 4 scores.

Those scores are not yet probabilities. They are raw preferences. Then you apply `softmax`, which converts the scores into numbers between 0 and 1 that add up to 1, giving you a routing distribution. If expert 2 gets `0.55`, expert 0 gets `0.30`, expert 1 gets `0.10`, and expert 3 gets `0.05`, then for top-2 routing you keep experts 2 and 0, ignore the rest, run the token through those two experts, and combine their outputs with weights based on the routing scores. In English, this token mostly wants expert 2, it also wants some help from expert 0, and the other experts do not get involved.

Now the equation, because you have earned it. Let:

- `x` be the token hidden state
- `W_router` be the router's weights
- `N` be the number of experts
- `p` be the routing probabilities over those experts

Then the router computes:

```text
p = softmax(x W_router)
```

What each symbol is:

- `x` is one token's current representation
- `W_router` is the learned matrix that turns that representation into expert scores
- `x W_router` gives one score per expert
- `softmax(...)` turns those scores into probabilities that sum to 1

Then if the router picks experts `i` and `j`, the output is:

```text
y = p_i * Expert_i(x) + p_j * Expert_j(x)
```

Run the token through expert `i`, run it through expert `j`, then mix those two outputs using the router's confidence. Top-1 routing removes the second term: one expert wins and everyone else sits out. Top-2 routing keeps two experts, costs more compute than top-1, but is usually more stable and often gives better quality than people expect from "just one more expert."

The mental model I ended up with, after enough debugging, is that the router behaves like a dispatcher who only learned the job by watching which calls did not result in a complaint. There is no manager telling it the right call to make. There is only the loss yelling at it after the fact. That changes how you read a bad routing histogram. It is not the router being lazy. It is the router doing the only thing the gradient ever rewarded it for.

### Why Experts Help At All

A dense FFN must compress many patterns into one set of weights, whereas an MoE layer can store more patterns because different experts can specialize. This is the trick behind statements like "a model has 47B parameters but only 12B active per token." The full checkpoint contains a large amount of learned knowledge, but for any one token, only a smaller active slice runs.

MoE therefore gives you a new knob. Total parameters set how much knowledge the model can store. Active parameters set how much compute each token pays for. Dense models tie those two quantities together; MoE partly separates them. That separation is why frontier models care about this design. Figure 11.1 illustrates the router collapse failure, and Figure 11.2 shows how to read expert utilization to diagnose whether routing is healthy.

![Figure 11.1. Mixture-of-experts works only when the router distributes traffic across the expert pool; collapse sends most tokens to one expert and wastes the rest.](figures/fig_moe_router_collapse.png)

![Figure 11.2. Expert utilization tells you whether the MoE layer is healthy: balanced routing keeps the full pool learning, while collapse quietly turns most experts into dead weight.](figures/fig_moe_utilization_histogram.png)

## Why It Matters

Without MoE, growing model capacity usually means growing per-token compute — every token pays the full bill. If you double the width of dense feedforward layers across the model, you also double a big chunk of the work done for every token at training time and inference time.

MoE breaks that link. You can increase total parameters by adding experts while keeping only top-1 or top-2 active, which means more storage of learned behavior without making every token expensive in the same way a dense expansion would. This matters for three reasons, and the third one is the one nobody warned me about.

### 1. It changes the scaling game

Dense scaling says: "Want more capability? Spend more compute every time." MoE says: "Want more capability? Add more specialists, but only wake a few." That is not free. Routing adds overhead, training gets more fragile, and communication gets harder in distributed systems. The economics can still work out well enough to matter.

### 2. It creates a new failure surface

A dense FFN does not have routing collapse because there is nowhere to route; every token goes through the same path. MoE can fail in ways dense models cannot. One expert absorbs nearly all traffic. Some experts become undertrained. Top-1 routing becomes brittle, expert assignments drift wildly during training, and load balancing loss fights task loss too hard and hurts quality. You do not really understand MoE until you watch it fail.

In that ablation work, the failure mode I had braced for was router collapse to a single expert. The failure I actually got first was the opposite. The router fanned traffic out almost perfectly evenly across all experts, which sounds healthy, except none of them were learning anything useful because no token had committed to any of them. Balanced is not the same as alive. That was a humbling Tuesday.

### 3. It explains why "parameter count" gets slippery

Once MoE enters the picture, "How big is the model?" stops being a single clean number. You now need at least three numbers: total parameters in the checkpoint, active parameters per token, and how many experts are actually doing useful work. A model with 40B total parameters and 12B active can behave very differently from a dense 12B model, even if they spend similar compute per token at runtime. A sparse 8x7B model does not behave like "56B dense." It stores more knowledge than a 7B dense model, but each token still only activates a small slice, and the active path is what matters.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/18_mixture-of-experts/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/18_mixture-of-experts/build.py --full

# The BREAK IT experiment:
python projects/18_mixture-of-experts/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 18 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
