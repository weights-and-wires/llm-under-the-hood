# Project 35: Your Architecture

## Hook

How do you know whether your architecture idea is real, rather than one lucky run on one friendly benchmark?

Most research mistakes do not come from failing to build something new. They come from fooling yourself about whether the thing works at all. I have written that sentence three times in different forms across drafts of this chapter, and it kept coming back because I do not know how to soften it without losing what it means.

## The Concept

By this point in the book, we have built enough of the stack to stop treating LLMs as sealed appliances. We have seen gradients move weights, attention route context, routers choose experts, fine-tuning improve one thing while quietly damaging another, quantization trade quality for size, and composition look clean in a diagram and messy in code.

Project 35 changes the job. Up to now, the book asked you to understand mechanisms other people already chose. Here you choose the mechanism.

That sounds freeing. Mostly it means the excuses disappear.

Imagine you run a cargo port. Earlier chapters taught you how ships move, how cranes lift containers, how fuel gets delivered, and how schedules stay on time. Now someone asks you to redesign the port so it can handle a new kind of cargo faster than the old design. You are not being asked to admire ports. You are being asked to change one rule of the system and prove that the whole place still works. That is architecture research.

An architecture is not only a model shape. It is a claim about structure and why that structure should improve something that matters. Maybe your claim is: specialists should share a stricter interface so they compose better; routers should look at uncertainty, then hidden states; privacy-preserving fine-tuning should keep a specialist useful without exposing sensitive data; fusion should happen in hidden-state space instead of parameter space; or experts should be selected with a budget-aware mechanism that cares about latency, then quality.

A research contribution is not "I changed the code." It is a claim, a minimal test, a measurement, and a failure mode caught before someone else catches it for you. Without one concrete thread this chapter turns into hand-waving, so we anchor it to a composability problem sitting right in the middle of Part V.

**Problem:** specialist fine-tuning often improves domain skill while damaging composability.

In plain English: you teach one model to become a strong code specialist, another to become a strong legal specialist, and another to become a strong multilingual specialist. Each one gets better at its own job. Then you try to assemble them into one system. The assembly is shaky. Routing gets worse, order matters, hidden states no longer fit together cleanly, one specialist barges into another specialist's lane. The pieces no longer share the same internal dialect.

**Hypothesis:** if you force each specialist to preserve a shared internal interface while it fine-tunes, later composition becomes more stable.

"Shared internal interface" is still vague until you pin it down. Think of three musicians improvising together. If each one practices alone and changes tuning, timing, and key signature however they like, each may sound better solo. Put them in the same room and the session turns to noise. The shared interface is the tuning system they agree to keep.

For models, that interface can mean things like: same hidden-state dimension `d_model`, same normalization style, same positional encoding, same frozen early layers, same calibration on a shared anchor dataset, same expectation about where information lives in the residual stream. You do not need the full internal representations to be identical. You need them similar enough that swapping or combining specialists does not destroy coordination.

I cut a 2,000-word version of this section that tried to do this properly with diagrams and walked-through equations. I reread the cut version a month later and the same line came out of me: I get it, but this is the chapter where you stop teaching me and start trusting me to think. The musicians stayed.

So here is the example idea: train specialists with their task loss, but add an **interface preservation loss**. In plain terms: improve on the domain, but do not wander too far from the base model on a shared anchor set.

The task loss is the voice coach saying "sound better at your solo." The interface preservation loss is the band leader saying "stay in key." If you only keep the band leader, nobody improves. If you only keep the voice coach, nobody plays together. You need both.

This metaphor stuck because I tried five worse ones first. Two of them are still in my notebook with crossed-out arrows.

Now the equation is earned. Let `L_task` mean the normal training loss for the specialist's domain, `L_interface` mean a penalty for drifting away from the base model's hidden states on anchor examples, and `lambda` mean how hard you push on preserving the interface. Then train with:

```text
L_total = L_task + lambda * L_interface
```

Nothing mystical happened there. The equation just compresses one rule into a line: get better, but pay for drifting too far. You can make `L_interface` concrete in several ways. The easiest version is mean squared error between hidden states from the base model and the specialist at selected layers on a shared anchor set — show both models the same neutral examples and charge the specialist every time its internal representation wanders too far from the base.

That is one architecture idea. Project 35 is bigger than that one idea. The point is to learn how to create, test, break, and publish your own.

## Why It Matters

Without a project like this, you stay in demo-land. Demo-land is where an idea looks good because the test was weak.

A weak test asks "did the metric go up on the thing I cared about?" A real test asks harder questions: what degraded while the metric went up? What happened when I added a new specialist later? What happened when I changed the order of composition? What happened when compute got tight? What happened when domains overlapped, when the router was noisy, when the specialist data was smaller than expected, when I tried to reproduce the result with a second seed?

That shift matters because architecture work is full of flattering accidents. Here are common ways people fool themselves. They compare against a weak baseline, which teaches almost nothing if your method wins. They optimize one metric and ignore damage elsewhere, so a code specialist that gains 4 points on code completion while losing 12 points everywhere else gets mistaken for progress. They test composition only in the order they trained, which is choreography rather than composability. They report one lucky run, even though a result that vanishes on the next seed was never solid. And they confuse bigger compute with better design, when the real contribution may simply be a larger budget.

I have caught myself doing four of these five at some point in the last two years of my own work. I am not above the list. Nobody is.

Part V has been driving toward one claim. A modular system is not proven modular because you assembled a few neat pieces once. It is proven modular when you can keep adding pieces later without wrecking the old ones.

That is why this final project matters. It moves you from "I can implement known systems" to "I can define a failure surface, build an intervention, and test whether the claim survives." That is a field contribution. It does not have to be huge. It does have to be honest, reproducible, and sharp enough that someone else can prove you wrong.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/35_your-architecture/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/35_your-architecture/build.py --full

# The BREAK IT experiment:
python projects/35_your-architecture/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 35 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
