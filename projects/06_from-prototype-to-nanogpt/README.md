# Project 6: From Prototype to nanoGPT

## Hook

Why does a cleaner implementation of "the same model" often train better, run faster, and break less often? If my blank-file GPT already works, why does nanoGPT look different in so many small places that seem cosmetic until they are not?

This is the moment where a lot of people get annoyed: you finally built a transformer yourself, then you open a reference implementation and it feels like someone quietly swapped your homemade bicycle for a shop-built racing bike and claimed they are both "just bikes." They are both bikes. But one was built to prove understanding, and the other was built to survive repeated use. This chapter is about learning to see that difference clearly.

Here is the opinion I want to commit to upfront, because I do not think it is said often enough: nanoGPT is shaped by scars, not by design. Every "clean" choice in that file is the residue of some earlier run that went sideways. You will read it better if you treat each section as evidence of a failure someone learned from.

## The Concept

Your Project 5 GPT and `nanoGPT` are like two kitchens that both produce soup.

In your kitchen, every pot, spoon, and cutting board is exactly where you left it. You know why each thing is there because you placed it yourself. It works. You can make dinner. But if three more people walk in and try to help, the room turns into a mess fast.

`nanoGPT` is the same kitchen after someone who has burned meals for years reorganizes it. The knives are still knives. The stove is still a stove. The recipe is still soup. But the counters are clear, the hot pans have designated landing spots, and the ingredients are staged in the order they are used. Nothing about that sounds like "new cooking theory." It sounds like tidiness. Then service starts, and suddenly that tidiness is the difference between calm and disaster.

That is the first big idea of this project: a production-shaped minimal GPT does not mostly differ by adding exotic algorithms. It differs by removing avoidable friction. Some of that friction is code structure. Some is data movement. Some is training hygiene. Some is numerical stability. And a few parts that look boring, especially residual connections and LayerNorm, are not polish at all. They are life support.

We built our first GPT to understand the machine. Now we read `nanoGPT` to understand what a machine looks like after someone has been forced to care about speed, stability, and repeatability. The mental model to carry through the whole chapter: your prototype answers "What are the essential parts of GPT?" and `nanoGPT` answers "What are the essential parts once real training runs enter the room?" Those are not the same question. Your Project 5 file is a sketch. `nanoGPT` is a field manual. A sketch is honest; it shows the structure. A field manual is also honest, but in a harsher way. It has been shaped by failure. That is why this chapter matters. You are not reading bigger code to feel impressed. You are reading it to see where repeated failure carved the code into its final shape.

The first time I tried to port my blank-file GPT toward nanoGPT, I assumed the gap was style. Naming. File organization. Maybe a small speedup somewhere. I was wrong. Most of the gap was about what happens when you run the same code a hundred times instead of three.

Before code, let's anchor the main differences in plain English:

**Embeddings**
An embedding is a learned table that turns token IDs into vectors, meaning lists of numbers that represent each token. Your prototype probably treated embeddings as "the first layer." `nanoGPT` treats them as one part of a system that also includes positional information, dropout, **initialization**, and sometimes weight tying.

**Transformer blocks**
A transformer block is the repeating unit of GPT: attention, feedforward network, normalization, and residual paths. In a first implementation, the block is easy to read as a sequence of operations. In `nanoGPT`, the same block is arranged to train reliably at depth.

**Output projection**
At the end, the model turns hidden states back into vocabulary logits, one score per possible next token. A logit is an unnormalized score before softmax turns scores into probabilities. Your prototype may use a separate linear layer for this. `nanoGPT` often ties this output matrix to the token embedding matrix so the model reads and writes using the same learned dictionary.

**Initialization**
Initialization means how you choose the starting values of the weights before training. Your first version may use PyTorch defaults or a rough normal distribution. `nanoGPT` makes more deliberate choices because the starting scale of activations affects whether training begins as a controlled signal or as noise.

**Forward pass**
The forward pass is running inputs through the network to get logits and loss. In your prototype, `forward()` probably does exactly what you think the math says. In `nanoGPT`, `forward()` also reflects practical concerns: shape checks, optional targets, efficient loss computation, and clearer interfaces for training versus generation.

**Generation loop**
Autoregressive generation means the model predicts one token, appends it, and repeats. Your prototype probably recomputes more than necessary and keeps the logic simple. `nanoGPT` still keeps it readable, but it is sharper about evaluation mode, context cropping, temperature, and top-k sampling.

And then there are the two quiet heroes of the chapter:

**Residual connection**
A residual connection adds the block input back to the block output: `x + F(x)`. If each layer is a person editing a document, the residual path keeps a photocopy of the previous version attached to the new one. Even if the editor makes a terrible change, the old content still survives. That matters during the backward pass too. The gradient, which is the number telling each parameter how to move to reduce loss, has a direct path backward through the addition.

**LayerNorm**
LayerNorm rescales activations so each token representation stays in a reasonable numeric range. If every layer is a musician in a long orchestra chain, LayerNorm is the sound engineer who keeps the volume from drifting wildly louder or quieter after each instrument passes the signal along. Without that engineer, small imbalances compound until the audio clips or disappears.

People often treat residuals and norms as architectural decorations. They are not. They are the parts that let depth exist. And depth is the whole point of transformers. Two layers can bluff their way through, but six layers begin asking whether your design choices are serious. That is why the BREAK IT section in this chapter matters more than the code tour. When you remove residuals and LayerNorm, you stop discussing architecture in the abstract. You watch the machine lose the ability to learn.

## Why It Matters

If you stop at your prototype, you risk learning the wrong lesson. You might think the core equations are all that matter, that any training loop that "works" is basically fine, that residuals and normalization are detail work, and that clean code and stable code are separate concerns. All of those beliefs fail the moment you try to train a deeper model, compare experiments, resume a run, or extend the code without fear.

This project matters because it teaches you to separate three categories that beginners often mix together:

**Category 1: True algorithmic necessities**
These are parts without which the model does not train or does not scale in depth. Residual connections and LayerNorm live here.

**Category 2: Engineering choices that change speed, clarity, and robustness**
These include how batches are built, when validation runs, how checkpoints are written, how generation hooks are handled, how optimizer parameters are grouped, and how loss computation is structured.

**Category 3: Style differences that are mostly about readability**
Some naming and file organization choices land here. They matter, but they matter less than the first two categories.

If you cannot tell which is which, then every line in a reference implementation looks equally mysterious, and you learn nothing. You either cargo-cult the code or reject it as overbuilt. By the end of this chapter, you should not admire `nanoGPT` from a distance. You should steal from it selectively, not because it is bigger, but because you understand what each stolen part protects.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/06_from-prototype-to-nanogpt/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/06_from-prototype-to-nanogpt/build.py --full

# The BREAK IT experiment:
python projects/06_from-prototype-to-nanogpt/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 6 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
