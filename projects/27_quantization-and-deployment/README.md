# Project 27: Quantization and Deployment

## Hook

Why does a model that barely fits on a 24GB GPU suddenly run on a laptop after one conversion step, and why does that same trick eventually turn good answers into polished nonsense? That is the confusion to start with. If the model learned meaning through millions or billions of carefully adjusted floating-point numbers, how can you slash those numbers down to 8 bits or 4 bits and still get sensible text? And if that works, why does 2-bit quantization so often cross a line where the model still sounds fluent but stops saying anything true or coherent?

## The Concept

Think of a trained model as a giant wall of tiny knobs, where each weight is one knob. Training spends a huge amount of compute nudging those knobs until the model predicts the next token well. In FP32, each knob stores its setting using 32 bits, which is 4 bytes. A weight might be `0.0187342` instead of just `0.02`.

Now imagine you need to move this wall of knobs into a smaller room. One option is to build a smaller wall from scratch. That is retraining or designing a smaller model. The other option is to keep the same wall but repaint every knob with fewer allowed positions. Instead of letting a knob land on nearly any decimal value, you force it into a small set of buckets. That is quantization. FP32 says, in effect, "this knob can sit almost anywhere." INT8 says, "this knob must choose one of 256 positions." INT4 says, "pick one of 16 positions." INT2 says, "pick one of 4 positions."

The whole trick works because models are more tolerant to roughness than people first expect. A trained model does not need every single weight to stay exact down to many decimal places. Many weights can be rounded and the overall computation still points in roughly the same direction. The logits change a bit, but not enough to break the answer. Until they do.

That is the heart of this chapter: quantization is controlled damage. You are compressing the model by forcing many precise values into fewer buckets, then measuring whether the model still behaves acceptably for your job.

The first time I sat with this idea seriously was during the Orion work on Apple Neural Engine kernels. The whole point of getting an 8.5x speedup on delta compilation through private ANEClient/ANECompiler APIs was that ANE only natively supports lower precisions. Quantization was not optional; it was the price of entry. I expected the hard part to be the math. The hard part was the bookkeeping. The math is two lines.

### A plain-English picture of what gets lost

Suppose you have a grayscale photo with millions of pixels. In full precision, each pixel might store a very exact brightness. If you reduce that image to 256 shades, it still looks fine. Reduce it to 16 shades, and edges get harsher, but you can still recognize faces. Reduce it to 4 shades, and you still see rough structure, but fine details disappear. That is what quantization does to model weights.

The model still keeps broad patterns first. Syntax is a broad pattern. Phrase structure too. The rhythm of language as a whole stays robust. Fine semantic distinctions are more fragile. That is why heavily quantized models often fail in a very specific way: they speak in sentences that sound plausible while saying the wrong thing. The grammar survives longer than the meaning.

I have stopped describing this politely to non-ML colleagues. The honest summary is that a two-bit model sounds like a confident person on day three of a bad flu. The words still come out. The sentences still finish. The content has come unstuck from reality.

### What a quantized weight actually is

Here is the intuitive version before the formula. A block of FP32 weights contains real numbers like `-0.91`, `0.03`, `1.27`, and `-0.44`. To store them as INT8, you choose a mapping from real values to integers. For example, in this block:

- `-1.28` maps to `-128`
- `0.0` maps to `0`
- `1.27` maps to `127`

Now each original number gets rounded to the nearest integer bucket. Later, during inference, you approximately reconstruct the original value by scaling the integer back into a float-like range. So quantization has two parts:

1. Compress the number into a low-bit code.
2. Decompress it just enough at runtime to do the math.

The low-bit value is not the real weight. It is a compact code plus a rule for turning that code back into an approximate weight.

### The First Equation, Earned

Now that the picture is clear, here is the common formula for simple linear quantization:

$$
q = \text{round}(x / s)
$$

$$
\hat{x} = q \cdot s
$$

What do the symbols mean?

- `x` is the original weight, stored in FP32.
- `q` is the quantized integer value, such as an INT8 or INT4 bucket index.
- `s` is the scale, which tells you how much real-number distance each bucket represents.
- `\hat{x}` is the reconstructed weight used during inference after dequantization.

Read it in plain English: divide the real weight by the scale, round it to the nearest integer bucket, and later multiply that bucket by the scale to get an approximate weight back. That is the entire move. The error comes from rounding:

$$
\text{error} = x - \hat{x}
$$

If the buckets are fine enough, the error stays small. If the buckets are too coarse, the model starts forgetting distinctions it needs.

### Why Scale Matters

If the model has weights between `-0.02` and `0.02`, you need tiny buckets. If it has weights between `-7.0` and `7.0`, you need wider buckets. One fixed global scale for the whole model is usually too crude, so good quantization methods split weights into groups, rows, columns, or blocks, and give each group its own scale. That is the same as saying: do not compress a whole movie with one brightness setting. Compress each scene according to its own range. This is why you will hear terms like per-tensor quantization (one scale for a whole tensor), per-channel quantization (one scale per output channel or row), and group-wise quantization (one scale per small block of weights). Smaller groups usually preserve quality better, but they also add bookkeeping overhead. Again: quantization is tradeoff after tradeoff.

## Why It Matters

Without quantization, deployment hits a wall fast. A model with 7 billion parameters at FP32 needs about:

$$
7\,\text{billion} \times 4\,\text{bytes} = 28\,\text{GB}
$$

That is just the weights. Not the optimizer states from training, not activations during training, not the **KV cache** (the runtime attention key/value store described in **Project 13: Fast Inference: The KV Cache**, which in deployment is a significant memory consumer alongside model weights). Just the weights. That means FP32 blocks many local deployments outright, FP16 cuts memory in half but still leaves large models expensive, INT8 cuts memory to one quarter of FP32, INT4 cuts it to one eighth, and INT2 goes smaller still but often pushes quality below the floor.

This matters for three reasons.

### 1. Deployment stops being theoretical

If your model only runs in a data center, you have one kind of product. If your model runs on a laptop, edge box, or phone, you have another. Quantization is one of the main reasons local inference exists at all.

### 2. Memory bandwidth becomes the bottleneck

People often assume inference is only about arithmetic speed. For large language models, memory movement is often the real tax. You keep pulling huge weight matrices from memory into the compute units, and lower-bit weights mean fewer bytes moved. Fewer bytes moved often means faster inference. So quantized models are often smaller AND faster. Not always, but often enough that this is part of the default deployment toolkit.

This was the part that genuinely surprised me when I started measuring on Apple Silicon. I had assumed quantization was a memory-pressure story. It is. But it is also a bandwidth story, and the bandwidth story is often the bigger one. The compute units sit idle waiting for weights. Fewer bytes means less waiting.

### 3. Quality becomes an engineering budget, not a binary state

The right question is not whether quantization is good or bad. It is: how much quality loss buys how much deployment gain? For one product, a 1-point benchmark drop is unacceptable. For another, an 8x size reduction with a small reasoning hit is the whole business model. Once you can measure that tradeoff, you stop speaking in slogans and start speaking in numbers.

This is the reframing that runs underneath every honest deployment conversation I have had since shipping Marunthagam to rural ASHA workers in Tamil Nadu. We had a hard memory ceiling because the device was whatever Android phone the worker already owned. Lower-bit models were not an optimization. They were the only models we could ship. The question was never "can we afford quantization." The question was always "how much can we afford to lose."

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/27_quantization-and-deployment/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/27_quantization-and-deployment/build.py --full

# The BREAK IT experiment:
python projects/27_quantization-and-deployment/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 27 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
