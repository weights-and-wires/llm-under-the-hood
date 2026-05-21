# Project 7: The Details That Matter

## Hook

Why does one transformer train cleanly, another wobble for hours, and a third refuse to extend past its training length even though the code looks almost the same?

This is the part people hand-wave away with words like "architecture choice," which is a nice way of hiding the fact that tiny-looking decisions change what every layer expects to receive. Change the normalization type, activation function, or positional encoding, and you are not tweaking style. You are changing the internal contract of the model.

From kernel-level inference work targeting Apple's Neural Engine through private compiler APIs, this lesson was unavoidable. The hardware did not care which architecture name was on the model card. It cared which numeric ranges the activations actually landed in. A norm switch that looked cosmetic at the PyTorch level changed the precision envelope downstream enough that the delta compiler had to be tuned differently. The phrase "same architecture" stopped meaning anything useful after a while.

## The Concept

**Project 6: From Prototype to nanoGPT** got us to the point where a transformer block was no longer magic. Attention mixes context. The feedforward network stores and transforms patterns. Residual connections keep information flowing. LayerNorm keeps training from flying off the rails.

Now we zoom in on the details that look boring right up until they break your model.

Think of a transformer layer as an assembly station in a factory line. Each station expects the incoming parts to arrive in a certain shape, size, and orientation, not because the station is picky, but because it was built and tuned around that expectation. Normalization decides the statistical "shape" of the hidden state. An activation function decides how the model bends and gates information after a linear transformation. A positional encoding scheme tells the model where each token sits in a sequence. Those three choices sit at the boundaries between major pieces of the model, which is exactly why they matter so much. They define the interface.

Here is the confusion most people have the first time they hear this: "If LayerNorm and **RMSNorm** both stabilize activations, why would swapping one for the other matter?" Because "stabilize" is too vague. A refrigerator and a freezer both keep food cold; that does not mean you can replace one with the other in a recipe without changing the result. LayerNorm and RMSNorm both rescale hidden states, but they do not produce the same distribution. Figure 7.1 illustrates that key difference.

![Figure 7.1. LayerNorm and RMSNorm both control magnitude, but only LayerNorm recenters the hidden state around zero; that difference changes what downstream layers learn to expect.](figures/fig_layernorm_vs_rmsnorm_shift.png)

**GELU**, **SwiGLU**, and ReLU² all add nonlinearity, but they do not let information through in the same way. Learned positional embeddings and **RoPE** both tell the model where tokens are, but one memorizes positions while the other encodes relative offsets in a way that survives longer sequences.

In one sentence: a transformer is not just a pile of layers. It is a stack of agreements about what kind of signal each layer will hand to the next.

### Normalization: resetting the signal

A hidden state is just a list of numbers representing what the model currently "thinks" each token means at this layer. After attention and the feedforward network, those numbers can drift. Some dimensions grow too large, some shrink, and some layers start seeing values in ranges they were not trained to handle well. Normalization is the cleanup crew at the boundary.

LayerNorm says: for each token's vector, subtract its mean and divide by its standard deviation, then optionally scale and shift it with learned parameters. In plain English: center the vector around zero, then stretch or squeeze it so its spread is controlled.

RMSNorm says: skip the centering step — just divide by the root mean square, which is a measure of overall magnitude. In plain English: do not force the average to zero; only control the size.

That sounds minor. It is not minor. LayerNorm cares about both offset and scale. RMSNorm cares mostly about scale. So a layer trained after RMSNorm may expect a non-zero mean in certain dimensions, while a layer trained after LayerNorm may expect those means to be scrubbed away. That is why mixing them can produce a sharp seam in the network. The downstream layers were shaped by one kind of signal and suddenly receive another.

Across a long set of fine-tuning experiments I ran at scale, the single most common "this should have been easy" failure was a norm mismatch between two pretrained components that the user assumed were interchangeable. Same `d_model`. Same head count. Different distribution at the seam. The fusion would compile cleanly and then quietly produce worse loss than either component alone.

### Activation functions: the shape of response

A feedforward network is not just two matrix multiplications. Without a nonlinearity in the middle, those two matrices collapse into one, and the network loses expressive range. The activation function is the bend in the pipe. You pass numbers through it, and it changes which signals get amplified, softened, or shut down.

ReLU keeps positive numbers and zeros out negative ones. ReLU² keeps positives, squares them, and zeros out negatives. GELU lets values through smoothly, with small values getting partial credit instead of a hard cutoff. SwiGLU gates one stream with another: one projection carries content, and another projection decides how much of that content passes through.

If you want an analogy: ReLU is a binary door, open or closed. GELU is a soft door with a spring; it opens gradually. SwiGLU is a security gate with two checks: one path carries the package, the other decides whether the package gets through. These choices affect gradient flow, training stability, and final loss, not by huge dramatic amounts every time, but by consistent margins that accumulate over billions of tokens in real systems.

### Positional encoding: putting order back into a bag of tokens

Attention by itself has no built-in sense of order. Hand a model the tokens `["dog", "bites", "man"]` and `["man", "bites", "dog"]`. The set of tokens is the same, but the meaning is not, and without position information the model cannot tell the difference.

Learned positional embeddings handle this by giving every position index its own learned vector. Token embedding plus position embedding gives the model "word meaning at position 17." That works inside the training range, but it is like memorizing seats in one theater — move to a stadium with more rows, and the seat map runs out.

RoPE, short for Rotary Position Embeddings, handles position differently. Instead of attaching a separate learned vector for each position, it rotates pairs of hidden dimensions by an angle that depends on the token's position. To make that physical: imagine each pair of numbers in the hidden state as a little arrow on graph paper. Position 0 points the arrow one way. Position 1 rotates it slightly. Position 100 rotates it more. When queries and keys compare themselves, the angle between them carries relative position information. The key insight is that rotation is relative. If token A is at angle θ and token B is at angle 2θ, their dot product encodes not their absolute positions but the angle between them, which is exactly the relative distance. That means the same attention pattern works whether those tokens appear at positions 5 and 10, or 105 and 110.

The positional signal becomes part of the geometry of the vector space, not a lookup table of position IDs. That is why RoPE often extrapolates beyond training length better than learned positional embeddings. Not perfectly, but better.

I expected RoPE to feel exotic the first time I implemented it. It does not. It feels mechanical. A rotation matrix on pairs of dimensions, indexed by position. The mystique evaporates as soon as you write the two-line rotation by hand. What does not evaporate is how much careful frequency scheduling matters, and that is the part I still re-read references on every time.

## Why It Matters

If you only care whether the loss goes down, these details look decorative. If you care whether the model trains reliably, serves quickly, extends to longer sequences, or can be composed with another model later, these details are where the real engineering starts.

### Why normalization type matters

Without normalization, deeper models drift. One layer pushes activations a bit wider, the next layer amplifies that, and a few blocks later you do not have a stable signal — you have a distribution problem. Residual connections help information move through the network. Normalization keeps that moving information statistically sane.

LayerNorm and RMSNorm do this in different ways, and that difference shapes what downstream layers learn to expect. If two independently trained components do not agree on hidden-state statistics, you cannot bolt them together safely just because the tensor shapes match. Shape compatibility is not enough; statistical compatibility matters too.

### Why activation choice matters

You can get a model to train with several activation functions, which tempts people to think activations are interchangeable. They are not. The activation controls how easy it is for the model to represent small changes, how sharply it gates signals, and how gradients move backward through the feedforward network. These changes often show up as small but repeatable differences in final validation loss. In production work, "small but repeatable" is not a footnote. It is the whole job.

### Why positional encoding choice matters

Learned positional embeddings bake in a fixed table of position vectors. That is simple, and it is also brittle if you want longer contexts later. RoPE changes the attention calculation itself, making relative position survive in the dot products between queries and keys. That gives you a path to longer contexts without retraining the position table from scratch. If you want to understand why some models stretch from 2K to 8K or 32K context with careful fine-tuning and others faceplant, this is one of the first places to look.

### Why instrumentation matters

Loss tells you whether the whole machine is getting better. It does not tell you what each layer is doing internally. Instrumenting hidden state statistics gives you X-ray vision. Mean, variance, min, max, **kurtosis**: these are not random numbers for a dashboard. They tell you whether a layer boundary keeps the signal centered, whether variance explodes or collapses, whether outliers dominate, and whether a mixed architecture introduces a seam. This project is where you stop treating the model as one giant black box and start looking at the distribution of the blood flowing through it.

The strongest opinion I will commit to in this chapter: per-layer activation statistics should be on by default in any serious training script. Not a debugging feature you bolt on after something fails. A first-class output of every run. I have lost more hours to "I wish I had logged that" than to any one architectural mistake.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/07_the-details-that-matter/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/07_the-details-that-matter/build.py --full

# The BREAK IT experiment:
python projects/07_the-details-that-matter/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 7 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
