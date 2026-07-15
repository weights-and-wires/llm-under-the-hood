# Project 33: Fusing Independently Trained Specialists

## Hook

Why can three separately trained specialists sometimes combine into one system that beats all three, yet the same recipe falls apart the moment those specialists come from different starting models? A good code model and a good medical model are not enough. Their hidden states have to mean roughly the same thing. Without that, the router is doing traffic control with three street maps drawn in different coordinate systems.

This was an awkward chapter to settle. The first version sold the success case too hard. The real value of the project is the failure case. The success says "this can work." The failure says "here is the exact condition that has to hold for it to work at all." I cut the optimistic framing and left the boundary condition standing.

## The Concept

Start with an analogy that behaves like the system.

Picture three translators at one desk. One handles Python code, one handles medical notes, one handles legal contracts. A clerk skims the incoming page and decides who gets it. The setup works if all three translators learned language from the same textbook and then specialized. The clerk is not reading their minds, but the shared training leaves enough common structure that routing stays legible.

Now wreck that assumption. Each translator learned from a different textbook, with different notation habits, different grammar instincts, different shortcuts. The desk still looks tidy from the outside. Inside, the clerk has no stable basis for comparison. "This feels medical" only makes sense relative to some internal frame, and the specialists no longer share one.

I think a lot of the modular-AI literature glosses past this part too quickly. The architectures look composable on a slide. The internal coordinate problem only shows up when you try to actually wire two independently trained models together and watch the router thrash. That moment, where the system is technically correct and behaviorally broken, is the experience this chapter is built around.

That is the whole project. You take one pretrained model, freeze the early layers so every specialist keeps the same trunk, then train three specialists on three domains: code, medical text, and legal text. Each specialist keeps the same early shared representation and adapts later layers to its domain.

Then you build a router. A router is a small model that looks at the shared hidden state and decides how much weight to assign to each specialist. If the input smells like code, push weight toward the code expert. If it smells like a discharge summary, push toward the medical expert. This is not standard Mixture of Experts, where one large model grows experts and routing together during training. It is closer to after-market composition: you train specialists separately, keep the interface controlled, and ask whether they still compose afterward.

That question matters because it changes what "building a model" means. The old picture is one giant model. The newer picture is modular: keep the shared backbone steady, train specialists independently, connect them with routing. That only works if the modules agree on the signals passing between them. If two specialists start from the same base and preserve the same early layers, their hidden states still live in roughly the same internal coordinate system. If they come from unrelated pretrained models, that shared coordinate system disappears. The router then needs more than three competent specialists. It needs specialists that speak a compatible internal language. Figure 18.1 diagrams the architecture that makes this work.

The KALAVAI cooperative LoRA-fusion experiments hammered this lesson home across long training sweeps spanning several model families. The runs where contributors started from the same pretrained base and only their LoRA adapters varied composed cleanly. The runs where contributors started from different bases needed a lot more scaffolding to even produce a coherent fused output, and they often still underperformed the single best specialist. The shared-foundation requirement was not a paper detail. It was the difference between a system that worked and a system that did not.

![Figure 18.1. Fusion only works when the router reads a genuinely shared interface: the trunk stays common, specialists diverge later, and routing happens from that shared hidden-state boundary.](figures/fig_specialist_fusion_router.png)

## Why It Matters

The payoff is modularity you can test: add a specialist without retraining everything, isolate failures by domain, and ask whether composition beats any single expert. But there is a hard constraint hiding underneath. Composition depends on shared initialization.

"Initialization" means the starting parameter values of the model before fine-tuning. Shared initialization means the specialists all began from the same pretrained checkpoint: they were born with the same internal geometry. Without that, the router is not just choosing between specialists. It is standing between incompatible worlds.

Honestly, I think this is the single most under-discussed constraint in the entire modular-AI conversation. People will compare seven router architectures and skip the question of whether the experts they are routing to share a basis at all. The router design barely matters if the inputs to it do not live in the same space.

**Project 32: Layer Freezing and Transfer** asked where specialization starts when you freeze layers. This chapter asks a sharper question: once specialists exist, can they combine into something better than any one of them alone? And then it asks the question that matters more: what exact condition makes that combination possible? The answer is not "have a good router." The answer is "have a shared foundation first."

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/33_fusing-independently-trained-specialists/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/33_fusing-independently-trained-specialists/build.py --full

# The BREAK IT experiment:
python projects/33_fusing-independently-trained-specialists/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 32 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
