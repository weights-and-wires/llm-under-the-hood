# Project 16: Long-Context Extension (RoPE, YaRN, NTK-Aware)

## Hook

Your model from **Project 7: The Details That Matter** was trained on 1024-token contexts. You hand it a 4096-token document, ask it the same question you would have asked of a shorter document, and the output comes back as garbage. Repeated tokens, broken grammar, occasional words from another language, a kind of staring-into-static feel that a smaller context would never produce.

Nothing about the architecture changed. The weights are the same weights. The tokenizer is the same tokenizer. The math underneath rotary position embedding (the technique your model already uses to encode where each token sits) is the same math. So why does the model fall apart the moment the document gets long enough?

The first time I watched this happen in the wild was during a multi-corpus run at varying context lengths. The model that had been clean at 2k context produced something that looked, on the screen, like a transmission from a radio station drifting off frequency. I sat there for a few seconds before I realized the words on the page were not English anymore. They were not anything anymore.

This chapter answers what is going on, then walks through three fixes (position interpolation, NTK-aware scaling, and YaRN) that pull the model back into coherence at four times its training length.

## The Concept

Rotary position embedding, or RoPE, was introduced in **Project 7: The Details That Matter** and used quietly in every chapter since. The short version is this: instead of adding a position vector to the token embedding like the original transformer did, RoPE rotates each pair of dimensions inside the query and key vectors by an angle that depends on the token's position. Token at position 3 gets a small rotation. Token at position 17 gets a larger rotation. The dot product between a query at one position and a key at another now depends on the difference of their angles, which is the difference of their positions. That is how the model "knows" where things sit relative to each other.

Picture a clock face. The hour hand sweeps slowly. The minute hand sweeps fast. The second hand sweeps faster still. Each hand encodes time at a different scale: the second hand is good for telling apart adjacent ticks, the hour hand is good for telling apart whole hours.

RoPE does the same thing inside the hidden state. Each pair of dimensions inside the query and key vectors gets its own rotation speed. Some pairs rotate fast (many full turns even within a short context), and those pairs are the second hand, sensitive to nearby positions. Other pairs rotate slow (barely a fraction of a turn across the whole training window), and those pairs are the hour hand, sensitive to long-range structure.

This analogy took three tries to land. The first version used gears, which was technically more accurate but lost everyone. The second used radio frequencies, which lost everyone who had not worked in RF. The clock landed because every reader has stared at one.

The speeds come from a single hyperparameter: the **base frequency**, usually written as 10000. For a hidden state of dimension d, the rotation speed for the i-th pair is `1 / base^(2i/d)`. The first pair rotates fast, the last pair rotates slow, and the speeds slot in between as a smooth geometric series. This is the part to lock into your head before going further. RoPE is a clock with d/2 hands, each running at its own speed, all keyed off one base frequency.

Now the failure mode. During training at sequence length 1024, the fastest hands made many full turns and the model has seen every possible orientation for them. But the slowest hands barely moved. Across 1024 positions, the slowest pair might rotate through only a small arc of the unit circle. The model learned what queries and keys mean when the slow hands sit inside that arc. It has never seen them anywhere else.

Now feed the same model a 4096-token context. Suddenly the slow hands rotate four times farther than they did in training. They sweep into angles that no training example ever produced. The dot product between a query at position 3000 and a key at position 4000 depends on slow-hand angles that look, to the model, like noise.

**Extrapolation** is the word for this: pushing inputs past the range the model was trained on. RoPE extrapolates badly. The model has memorized a band of the clock face, and outside that band it has no idea what to do.

Three fixes. Each one is a different way of keeping the slow hands inside the trained band even when the context grows.

The first is **position interpolation**, or PI. Take every position index and divide it by the extension factor k. If your model was trained at length 1024 and you want it to handle length 4096, set k = 4 and remap position 4096 to position 1024, position 2048 to position 512, position 1 to position 0.25. The clock still runs through the same arc it always ran through; you have just stretched the timeline.

Every rotation angle the model now sees is an angle it has already seen in training. The cost is resolution. The second hand, which used to distinguish position 3 from position 4, now has to distinguish position 3 from position 3.25. Adjacent tokens get rotation angles that are four times closer together than the training distribution. That hurts fine-grained position discrimination, but it does not break the model the way raw extrapolation does.

The second fix is **NTK-aware scaling**. This one is sneakier. The intuition is: fast hands are fine (they have already seen every angle), so let them keep rotating fast. Only the slow hands need help. Instead of rescaling all positions uniformly the way PI does, NTK-aware scaling adjusts the base frequency itself so that the slow hands get stretched while the fast hands stay roughly the same. Formally, you replace the base 10000 with `base * k^(d/(d-2))`. That choice of exponent, `d/(d-2)`, is not magic; it is chosen so that the slowest pair gets multiplied by exactly k while the fastest pair barely shifts. Fast pairs interpolate normally; slow pairs extrapolate, but they extrapolate inside a band the model can still parse because the per-step rotation is now smaller. The result is a clock where the second hand still distinguishes adjacent positions, but the hour hand now sweeps an angle range similar to what it saw in training.

The third fix is **YaRN**, short for "Yet another RoPE extensioN method." YaRN is NTK-aware scaling plus a temperature adjustment on the attention logits. **Attention temperature** is the divisor inside the softmax, normally `sqrt(d)` for a head of dimension d, which keeps the softmax distribution stable as the head dimension grows. The problem at long context is that even after NTK-aware fixes the rotation geometry, the attention distribution drifts. Each query is now comparing itself to many more keys, the dot products spread out, and the softmax becomes sharper than it was at training time. The model attends to a single key per query instead of mixing several, and the attended key is often the wrong one. YaRN compensates by dividing the dot products by `sqrt(d) * temperature` instead of just `sqrt(d)`, where temperature is a function of the extension factor. A typical choice is `temperature = 0.1 * log(k) + 1`. The clock fix and the brightness fix run together, and the model holds together at longer context lengths than either fix alone.

A short analogy that ties it together. A clock trained to read 1 through 12 hours cannot read 1 through 48 hours. PI compresses the dial: the whole 48-hour range gets squeezed into the 12-hour arc the clock has seen. NTK-aware stretches the slow hand: the hour hand learns to sweep through more hours per turn, while the minute hand keeps its old pace. YaRN does the NTK trick and also turns down the brightness on the contrast knob so the eye does not get confused by sharper attention than it was trained for. None of these methods retrain the model from scratch. They are angle reparameterizations that ride on top of an existing trained model, and the best ones are surgical enough that even without any further fine-tuning the model stays coherent at four times its training length.

One more term before the build. The **needle-in-a-haystack** test is a benchmark introduced by Greg Kamradt and adopted by Anthropic and others. Hide a single odd fact ("the secret number is 1729") inside a long, otherwise unrelated document, ask the model to retrieve it, and measure success as a function of where the fact was hidden and how long the surrounding document is. A model that handles long context well retrieves the needle from any position. A model that does not handle long context well retrieves it from the start of the document and forgets the middle.

I have to register a strong opinion here, because the needle test has become so dominant that it now distorts the field. Passing needle-in-a-haystack does not mean a model can reason across a long document. It means a model can retrieve a single unusual string from a long document, which is much easier than what users actually want. I have seen models that ace 128k needle and still lose the plot of a 12k contract review. Use the needle test as a coverage check, not as proof of long-context reasoning. It is a coarse instrument that the field treats as a microscope.

## Why It Matters

Almost every product use case for an LLM has the same wish: longer context. Summarize this hundred-page contract. Answer questions about this codebase. Read this medical chart and look for anything odd.

Pretraining at one million tokens is enormously expensive, both because the compute scales quadratically with sequence length and because high-quality long documents are rare. The dominant approach in the field is to pretrain at a short length, then extend the trained model to a longer length using one of the techniques in this chapter, optionally followed by a short fine-tuning pass on long documents. Llama 2 was trained at 4k context and extended to 32k and 100k by Code Llama and Llama 2 Long. Yi was trained at 4k and extended to 200k. GPT-4 turbo went from 8k to 128k. The technique is the same template: RoPE rescale, brief fine-tune, ship.

If your model from **Project 7: The Details That Matter** breaks the moment a real document arrives, that limit is not a fundamental property of the architecture. It is a knob, and this chapter teaches you how to turn it. The same skill transfers to every open-weights model on the Hugging Face Hub. Llama, Mistral, Qwen, Yi, and DeepSeek all ship with RoPE; almost all of them ship with extension methods bolted on; and almost all of those methods are one of the three this chapter implements.

There is a second reason this chapter matters that surfaces only when you start running it. The **KV cache** from **Project 13: Fast Inference: The KV Cache** stores the rotated keys and values for every token in the context window. If you change the way RoPE applies its rotations partway through a session, you have to either recompute the entire cache or work out how the new rotation rule interacts with the cached keys. Most extension methods are applied at inference time, so the cache must be rebuilt under the new rule.

This chapter is a quiet prerequisite for serving long-context inference. **Project 15: Grouped Query Attention** reduces the cache size per token; long-context extension makes more tokens fit in cache to begin with. The two combine, and you will need both for any production long-context system.

I learned this the slow way. The first time I bolted YaRN onto a model with an existing KV cache pipeline, I forgot the cache was holding keys rotated under the old rule. The model produced clean output for the first 1024 tokens, then started drifting, then produced output that looked like the failure mode from the Hook. I spent two hours convinced YaRN was broken before I realized I had two RoPE rules running simultaneously on the same sequence.

There is a third reason, less practical but worth naming. Watching what specifically breaks when a model extrapolates past its training length teaches you something about what learning is. The model did not learn "positions" in some abstract sense. It learned a specific arc of the unit circle for each pair of dimensions, and outside that arc it has nothing to say. Most failures of generalization in deep learning look like this when you zoom in close enough. They are not failures of intelligence. They are failures of coverage.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/16_long-context-extension-rope-yarn-ntk-aware/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/16_long-context-extension-rope-yarn-ntk-aware/build.py --full

# The BREAK IT experiment:
python projects/16_long-context-extension-rope-yarn-ntk-aware/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 16 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
