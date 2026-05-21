# Project 8: Flash Attention and Tiled Kernels

## Hook

Why does the hand-written attention from **Project 4: Attention From Scratch** run out of memory on a 4K context while PyTorch's `F.scaled_dot_product_attention` handles the same input on the same GPU without breaking a sweat? Same math, same hardware, same sequence length, and yet one crashes and one does not. The textbook answer is "Flash Attention is faster," which is true and also useless if you cannot see why.

The first time I hit this wall I assumed it was a driver bug. It was not. The model was fine. The framework was fine. One tensor was the problem, and once you can name it, the fix is obvious.

By the end of this chapter you will know the exact tensor that vanishes between the two versions, you will build a memory-efficient attention from scratch in pure PyTorch, and you will watch your peak memory drop from quadratic in sequence length to linear without changing a single line of the model that calls it.

## The Concept

Picture a chef working on a tiny stove. The pan holds one cup of liquid at a time. The chef has a recipe that, in the original instructions, asks you to mix twenty cups together and then heat the whole batch. If you only have a one-cup pan, you have two choices. You can buy a bigger pan. That is the GPU-with-more-memory option, and it stops working as soon as the recipe scales up again. Or you can rewrite the recipe to work in small batches: a cup at a time, with a running total in a separate bowl, never holding more than a cup in the pan at once. That is Flash Attention. The pan is your GPU's fast on-chip memory. The bowl with the running total is the accumulator. The cup at a time is a tile.

The analogy that finally made tiled attention click for me did not involve a chef. It came from Orion, where we were programming the Apple Neural Engine directly because the high-level framework refused to honor our memory budget. The ANE has a small fast region and a large slow one, and every algorithm worth running on it eventually becomes a question of "what stays in fast memory, what gets streamed past it, and what never needs to be materialized at all." Flash Attention is the same question, asked of a GPU.

Attention asks the model, for every position in a sequence, to compare itself against every other position and decide who matters. With a sequence of length N, that comparison produces an N-by-N table of scores. At N equal to 4096, that table holds about 16.8 million entries per attention head, and the model has several heads, several layers, and a batch of sequences. The table is the pan that is too big. Materializing it (actually allocating that N-by-N tensor in memory and writing every score into it) is the move that breaks at long context.

Flash Attention's central idea is that you never have to materialize the table. The final output of an attention layer is a weighted sum of value vectors, where the weights come from a softmax over the score table. You can compute that final output in pieces. Take a small block of rows from the query, a small block of columns from the key, compute the partial scores for just that tile, fold them into a running output, throw the tile away, and move on to the next one. The full table is never written. The pan stays small. The result is identical.

The trick that makes this possible is something called an **online softmax**, also called a **streaming softmax**. A standard softmax over N numbers needs to see all N numbers before it can produce any output. It computes the maximum of the N numbers (for numerical stability), subtracts that maximum from every number, exponentiates each result, sums the exponentials, and divides each by that sum. Every one of those steps depends on all N numbers being available at once. That is exactly the dependency we want to break.

The online version watches the numbers come in one block at a time and keeps two running statistics: the running maximum seen so far, and the running sum of exponentials, rescaled to that maximum. When a new block arrives, you compute its own local maximum and local exponential sum, compare its maximum to the running maximum, and rescale the older running statistics if the new block's maximum is larger. The output you accumulate uses the same rescaling. By the end of the last block, the running statistics give you exactly the same answer as the all-at-once version, to the bit. No approximation. No shortcut. Just a different order of operations that never needs to hold the full vector in memory.

Two terms worth defining before they appear in code. A **tile** is a small block of a larger tensor, sized to fit comfortably in a fast memory region. On a real GPU, that fast region is the on-chip SRAM, perhaps 100 kilobytes per streaming multiprocessor. On the conceptual version we will build in pure PyTorch, the "fast region" is whatever amount of GPU global memory you have decided to limit yourself to. A **fused kernel** is a single computation that combines what would otherwise be several separate passes through memory. Instead of computing scores, writing them out, reading them back, applying softmax, writing the softmax back, reading it again, and multiplying by values, a fused kernel does all of that on one tile while the tile is still in fast memory. The win is not only less memory. It is also fewer round trips between fast on-chip memory and slow off-chip memory, which is where most of a GPU's wall-clock time goes during attention.

The original Flash Attention paper, by Tri Dao and colleagues in 2022, calls this property **IO-aware**: the algorithm is designed around the cost of moving data between memory tiers, not around the cost of arithmetic. Attention is bandwidth-bound at long context lengths, not compute-bound. Reducing memory traffic is the actual lever. The CUDA kernel in the production library is tightly tuned to the specific sizes of those memory tiers on real hardware, which is why writing it in CUDA pays for itself with another sizable speedup on top of the algorithmic win. But the algorithmic win is what we are after in this chapter, because the algorithmic win is what survives if you are reading this on a different accelerator three years from now.

I learned this lesson the slow way on low-level ANE work. We spent weeks trying to make CoreML respect a memory budget the hardware's spec sheet said it could hit. CoreML refused. So we wrote against the private ANEClient and ANECompiler APIs and shaped the kernel around the memory hierarchy ourselves. The headline number was a large delta-compile speedup, but the real lesson was about who owns the memory plan. If the framework owns it, you live with whatever the framework decided. If you own it, you can do exactly the kind of tiling this chapter is about.

A small note on what this chapter does not promise. We will not match the production CUDA kernel's wall-clock numbers. The pure-PyTorch tiled version you will build is meant to be readable, correct, and clearly memory-efficient. The speed win at moderate sequence lengths is real but modest; the memory win is dramatic. Speed parity with `flash-attn` requires CUDA, and CUDA is a different book. What you will have at the end of this project is the algorithm in your hands, in code you can read line by line, and a working understanding of why the production version is shaped the way it is.

I want to flag a pet peeve here. Most explanations of Flash Attention I have read open with the CUDA kernel and leave the algorithm implicit. That order is backwards. Once you understand the algorithm, the kernel is straightforward engineering. Without the algorithm, the kernel reads like incantations.

## Why It Matters

The reason this chapter exists where it does is that every chapter after it assumes long context as a baseline. Without tiled attention, the rest of the book runs into a wall at sequence length somewhere between 2K and 4K on most consumer GPUs. With tiled attention, the wall moves out by a factor of ten or more, and the projects that depend on long context become reachable on the same hardware.

The hand-written attention from **Project 4: Attention From Scratch** allocates a tensor of shape `(batch, heads, seq, seq)` to hold the scores. At sequence length 1024, with 8 heads and batch size 8 in fp16, that tensor is roughly 128 megabytes per layer. At sequence length 4096 with the same other settings, the same tensor balloons to 2 gigabytes per layer. A 6-layer model now wants 12 gigabytes for nothing but score tensors, before any of the actual model parameters, activations, gradients, or optimizer state. On an 8 GB consumer card, the run dies before it begins. The crash is not subtle. PyTorch throws an out-of-memory error and tells you exactly which allocation failed. The crash is also not the model's fault. The model's parameters fit. The forward activations fit. The single thing that does not fit is the materialized score table.

This was the specific failure mode that bit us hardest during fine-tuning experiments at multi-billion-parameter scale. The model and the optimizer fit. The activations fit. The attention scores tensor at the context lengths we wanted did not, and the failure was on a 4090, not on something exotic. A long tail of out-of-memory tracebacks taught me that "more memory" is the answer almost no one's hardware budget agrees to.

That same memory pressure is exactly the failure mode that the activation-stash investigation in **Project 11: Training Debugging — Spikes, NaNs, Profiling** identifies in its memory-forensics section. When the dominant allocation in your model is the attention scores tensor, the right fix is not "buy a bigger GPU." That scales poorly and runs out again at the next context length. The right fix is to stop materializing the tensor in the first place. That is what this chapter builds.

There is also a forward-pointing reason. **Project 13: Fast Inference: The KV Cache** introduces the inference-time optimization that caches keys and values across decoding steps, so that each new token only computes attention against the new query row against the full cached key and value blocks. That decoding-time computation is itself a tiled attention: a single query row tile against a long key-value column block. Once you have built the training-time tiled forward in this chapter, the inference-time variant in Project 13 is a small specialization of the same kernel, not a new mystery.

And there is a third reason worth naming. Once you can read the Flash Attention algorithm in pure PyTorch, you can read the Triton and CUDA versions in `flash-attn` and `xformers` and recognize the same shape underneath. The production code has more constants, more memory hints, more autotuning, and more low-level addressing math, but the algorithm is the algorithm. The mystique evaporates the moment you watch the running maximum get rescaled on a CPU tensor of size four.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/08_flash-attention-and-tiled-kernels/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/08_flash-attention-and-tiled-kernels/build.py --full

# The BREAK IT experiment:
python projects/08_flash-attention-and-tiled-kernels/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 8 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
