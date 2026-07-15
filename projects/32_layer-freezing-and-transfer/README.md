# Project 32: Layer Freezing and Transfer

## Hook

Why can you spend real money fine-tuning a model on your domain, get better results on that domain, and still somehow make the model worse at everything else it used to know? And why does the opposite extreme also fail, where you freeze almost the whole model, train for hours, and end up with something that barely changed at all? This is the engineering problem in transfer learning: how much of a pretrained model should stay untouched, and how much should adapt?

The first time I watched it happen in our own runs, I felt a small unfairness. The domain numbers were going up. The model genuinely looked better at the thing we trained it to do. And it was, quietly, losing other things we had not been watching. That is the actual texture of this chapter. The win and the cost share the same training step.

## The Concept

Picture a large office with many floors. The bottom floors handle generic work that every department needs: reading email, identifying names, spotting dates, recognizing punctuation, basic grammar, common patterns. The upper floors handle more specialized work: legal review, code style, medical jargon, finance phrasing, multilingual nuance.

A pretrained language model works a lot like that. Early layers tend to learn broad, reusable features. They notice local patterns, help the model tell where words begin and end, understand what phrases often travel together, how syntax hangs together, and how common token combinations tend to behave. Later layers tend to shape those general features toward a task or domain. They take "this looks like a function definition" or "this looks like a drug name" and turn it into behavior that matters for the next-token prediction in that setting.

I am stating this more confidently than the field actually deserves. The early-layers-are-general story is a useful guide, not a clean theorem. It bends with model size, with how much pretraining data covered the new domain, and with how aggressive the optimizer is during fine-tune. Treat it as a starting prior. The experiment in this chapter is what turns it into something you can actually trust for your own model.

Layer freezing means this: you take a pretrained model and tell some layers, "You are done learning. Do not move." Then you fine-tune only the remaining layers on new data. In plain English, you are saying: "Keep the general language machinery. Adapt only the parts most likely to need domain-specific change." That is the whole bet.

If the bet is right, you get three things at once: faster training because fewer parameters update, less forgetting because you leave more of the original model intact, and still enough adaptation to become better at your new domain. If the bet is wrong, one of two bad things happens.

At one extreme, you freeze nothing. Every parameter is free to drift, the model gets better at your niche domain, but it starts overwriting some of the general patterns that made it broadly useful. This is catastrophic forgetting (introduced in **Project 21: Fine-Tuning And Instruction Tuning**): the measurable price when domain fine-tuning gets too much permission to move shared internal structure. The first time an offline Tamil-language medical fine-tune hit this on small Gemma-class models, the clinical triage answers got tighter and the basic conversational fluency got noticeably more brittle on the same evaluation prompts the base model had handled fine. We were paying for clinical sharpness in plain-English coherence, and we did not see the bill until we added a non-clinical validation set.

At the other extreme, you freeze almost everything. The model has too little room to adapt, it still sounds like the original model because mostly it still is the original model, and it picks up a few surface habits but not enough to really learn the new domain. This chapter is about finding the useful middle.

### What "layer" means here

A transformer layer is one repeated processing block inside the model. In GPT-style models, each block usually contains:

- attention, which lets each token look at other tokens
- a feedforward network, which stores and transforms patterns
- normalization and residual connections, which keep training stable

If your model has 12 layers and you freeze the first 6, you are freezing the bottom half of that stack. The input still passes through those layers, they still compute outputs, they just stop changing during training. That distinction matters: frozen does not mean removed, it means read-only.

Most beginner explanations of freezing get this exactly wrong. They make it sound like a frozen layer becomes invisible to the forward pass, which then makes the speedup numbers feel like magic. The forward pass cost barely moves. What moves is the backward pass, the gradient storage, and the optimizer state. The intuition you want is "read-only memory, still wired in." Not "deleted."

### What transfer means here

Transfer learning means taking a model that already learned something useful from one dataset and adapting it to another job instead of starting from scratch. In this chapter, the transfer is from a general pretrained model to a domain-specific fine-tune target. That target might be:

- code
- medical text
- legal text
- another language
- support logs
- your company's documentation style

The reason transfer works at all is that the model's earlier layers are not learning "medical English" or "Python style" from zero. They are learning reusable machinery that many domains share. That is why freezing early layers is even plausible: if every layer were equally domain-specific, freezing early layers would be pointless.

### CKA: a way to measure "how much did this layer change?"

Now for the measurement tool that makes this chapter more than a hunch. CKA stands for Centered Kernel Alignment. That name sounds like a paper title because it is one. Ignore the name for a moment.

The intuition is simple. Suppose you run the same batch of text through the original pretrained model and the fine-tuned model. At each layer, you collect the hidden states (just the model's internal representation of the text at that point: a table of numbers where each token has a vector of features). Then you ask: "How similar are these two layers' internal representations?" If layer 3 in the original model and layer 3 in the fine-tuned model produce almost the same internal patterns on the same inputs, their CKA score is near 1.0. If they produce very different internal patterns, the score drops toward 0. So CKA gives you a map of representational drift.

The first time I plotted a real CKA curve on one of our cooperative fine-tune runs, I had a small embarrassing reaction. The smooth, gradual rise in drift I had pictured as you walk up the stack was not what showed up. The actual curve had a near-flat floor, a knee, and then a sharp climb. The knee landed almost exactly where we had set the freeze boundary. The diagnostic was not telling me something subtle. It was confirming that we had done the freezing correctly. That is a much more boring outcome than the one I had been bracing for, and a much more useful one for a paper.

That matters because layer freezing makes a very specific prediction. Below the freeze boundary, CKA should stay at 1.0 or extremely close, because those layers are frozen. Above the freeze boundary, CKA should drop, because those layers are adapting. That is more than a diagnostic. It is a composability signal. If early layers really stay stable across domains, then maybe those layers are reusable components. Maybe they can become shared infrastructure between specialists. That is one of the reasons this chapter sits in the composable architectures part of the book.

### The equation, now that you know what it is measuring

Here is a common linear CKA form:

```text
CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
```

Now the symbols have earned their place.

- \(X\) is the matrix of hidden states from one layer of the original model across many tokens or samples.
- \(Y\) is the matrix of hidden states from the matching layer of the fine-tuned model on the same inputs.
- \(Y^\top X\) means "compare all features in one representation against all features in the other."
- \(\|\cdot\|_F\) is the Frobenius norm, which you can think of as "take all numbers in the matrix, square them, add them up, then take the square root."
- The numerator gets large when the two representations line up well.
- The denominator rescales the score so you can compare layers fairly.

Do not worship the formula. Its job is modest. Take two clouds of activations, turn "these feel similar" into a number, that is enough.

Honestly, I think most introductions to CKA do it a disservice by leading with the equation. The equation looks more intimidating than the operation deserves. If you can compute a covariance matrix you already have everything you need. Lead with the question CKA is answering and the formula stops feeling like a hazing ritual.

## Why It Matters

Without layer freezing, you are stuck with a blunt instrument. You can fine-tune the whole model, or not fine-tune it. Real systems need more control than that.

### Why engineers freeze layers in practice

First: cost. Fine-tuning all layers means computing gradients and updates for all trainable parameters, which costs memory, time, and money. If you can freeze 50% of the stack and still get most of the gain, that is not a minor optimization. That is deployment reality. In long cooperative LoRA-fusion experiments, the gap between "every contributor fine-tunes their full base" and "every contributor fine-tunes a small adapter on a shared frozen base" was the gap between something interested volunteers could run on a consumer GPU and something they could not. Freezing decisions are infrastructure decisions.

Second: forgetting. A general pretrained model contains a wide range of behaviors. It can explain recipes, summarize logs, answer trivia, reason about text, and write code badly but competently enough to be useful. Full fine-tuning on narrow domain data can trade away some of that breadth. That trade can be worth it, but if you do not measure it, you are guessing.

Third: model composition. Part V of this book asks whether independently trained specialists can someday combine cleanly. That only has a chance of working if some parts of the network are more universal than others. Layer freezing gives you evidence. If early layers stay useful across domains and later layers absorb specialization, then the stack is not one inseparable blob. It has structure. Shared structure. That is the opening.

### What breaks if you ignore the tradeoff

If you freeze nothing, training is slower, memory use is higher, general capability can drift downward, and domain gain can come with hidden damage. If you freeze too much, training is cheaper, forgetting stays low, but the model barely adapts and domain metrics plateau early.

The engineering job is not "freeze as much as possible" and it is not "fine-tune everything." It is to find the point where domain gain is real, forgetting stays controlled, and the compute bill still makes sense. That sweet spot changes with model size, domain distance, dataset size, learning rate, training duration, whether you use LoRA or full-parameter updates, and how much the new domain overlaps with old knowledge.

In other words, there is no global answer. There is only the answer for your model on your data with your training budget, and the way you find it is by running the experiment described in the rest of this chapter and reading the curves. People who promise you a universal freeze ratio are usually selling something. This chapter gives you the measurement frame instead.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/32_layer-freezing-and-transfer/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/32_layer-freezing-and-transfer/build.py --full

# The BREAK IT experiment:
python projects/32_layer-freezing-and-transfer/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 31 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
