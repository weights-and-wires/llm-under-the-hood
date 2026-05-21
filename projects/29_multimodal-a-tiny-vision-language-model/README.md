# Project 29: Multimodal: A Tiny Vision-Language Model

## Hook

Your **Project 21: Fine-Tuning and Instruction Tuning** GPT cannot see. You can show it a photograph of a dog on a beach, and the model has no opinion about it, because the model only knows how to consume tokens, and a photograph is not a token. What does it take to make a text-only model caption an image?

This chapter builds the smallest possible vision-language model that actually works, and it does so as a proxy lab on one consumer GPU using CIFAR-style images. You will not need an A100. You will use 32-by-32 pictures, a tiny vision encoder, and the same small GPT you already trained.

The architecture you build here is exactly the architecture used by LLaVA, MiniGPT-4, and every other modern open vision-language model. The only thing that changes at scale is the size of the encoder and the size of the decoder. The mental model transfers without rewriting a single line of conceptual code.

A note on proxy honesty before we start. CIFAR-10 at 32-by-32 is a single-box proxy for what real VLMs do on 224-by-224 photographs of the open world. The wiring is identical; the resolution is not. I labored over this proxy choice while drafting the chapter, and I picked it because everything that matters about the architecture survives at small scale, while nothing that matters about the training loop requires hardware most readers do not own.

## The Concept

Take a translator and a writer. The writer only reads and produces text. They have spent years learning grammar, idiom, and how to finish sentences. They cannot look at pictures. The translator's job is different. They look at a picture, then describe it in a small handful of "words" that the writer can read. The writer trusts those words and continues the story from there.

That is the entire idea of a vision-language model. A vision encoder is the translator. A text decoder is the writer. A thin layer between them, called the projection, is the dictionary the translator uses to make sure the words come out in a vocabulary the writer recognizes.

The first version of this chapter buried this point. The blunter framing — the projection is the entire interesting part — came out of trying to explain the architecture to someone outside ML, where you cannot lean on the words "attention" or "embedding" to do the work. The chapter now opens with that framing.

A second analogy. Think of a text-only computer with no camera port. To add image input, you do not rebuild the computer. You bolt on a USB hub, and the hub converts a camera's signal into bytes the computer already knows how to read. The encoder is the camera. The projection is the USB hub. The decoder is the computer. Plug the hub in correctly, and the computer behaves as if it always had a camera.

Now the formal definitions, in plain language.

A **vision encoder** is a neural network whose input is an image and whose output is a list of vectors. Each vector represents one patch of the image. For a 32-by-32 image cut into 16 patches of 8-by-8 pixels, the encoder produces 16 vectors. Each vector has some dimension, say 192 numbers. In larger models the image is 224-by-224 and the patches are 14-by-14, which produces 256 vectors of dimension 768 or 1024. The shape changes; the idea does not.

When I first started thinking about image tokens on a Marunthagam pediatric-triage prototype that needed to read a photo of a malnutrition chart, I kept expecting image tokens to be doing something fundamentally different from text embeddings. They are not. They are vectors of the same dtype, the same dimensionality after projection, sitting in the same input slots. Watching the same on-device runtime dispatch a picture tensor and a word tensor through identical kernel paths drove this home. The hardware does not know one tensor is a picture and another is a word. It just shuffles bytes through buffers and dispatches kernels.

An **image token** is one of those output vectors. The name is deliberate. To the text decoder, the vector looks indistinguishable from a token embedding produced by looking up a word in a vocabulary table: same shape, same dtype, same place in the input sequence. The decoder does not know one is a picture and one is a word. It just sees vectors.

A **projection layer** is a small mapping that takes the encoder's output vectors and maps them into the decoder's embedding space. The encoder might produce 192-dimensional vectors. The decoder might expect 384-dimensional ones. The projection is a tiny multi-layer perceptron, often just two linear layers with a GELU between them, that does the dimension change and learns to put the image vectors in a region of the decoder's embedding space where they make sense as a prefix to text.

Two hundred thousand parameters. That is the whole multimodal capability. I will say that again in BREAK IT because it is the most underrated fact in modern VLM design.

The **decoder's embedding space** is the high-dimensional region where the decoder's normal word embeddings live. Every word in the decoder's vocabulary maps to a point in this space. Words that mean similar things end up close together. The projection's job is to put image tokens in this same neighborhood, so the decoder treats them as it would treat any other tokens at the start of a sentence.

A **vision-language model**, or VLM, is the full system (encoder, projection, decoder) wired together. The forward pass runs the encoder on an image to get image tokens, runs the projection to put them in the decoder's space, prepends them to a text prompt, and lets the decoder generate.

Two more terms to define before we build. **Captioning** is the task of producing a sentence that describes a picture: image goes in, sentence comes out. **VQA**, short for visual question answering, is a richer task. Image plus a text question goes in, an answer comes out. VQA is the multimodal version of instruction tuning. You already built the text-only version in **Project 21: Fine-Tuning and Instruction Tuning**, and the format is almost identical here.

![Figure 29.1. The vision-language model as three modules: a vision encoder turns the image into 16 image tokens, a projection layer maps them into the decoder's embedding space, and the text decoder consumes them as the first 16 positions of its context, then continues with text.](figures/fig_vlm_architecture.png)

The pattern is simple enough that it deserves a sentence in plain English. The encoder turns a picture into a small batch of vectors. The projection makes those vectors look like word embeddings. The decoder reads them as if they were words and writes a caption. The training job is to teach the projection what "looks like a word embedding" actually means for this encoder and this decoder.

## Why It Matters

If you cannot accept an image as input, an enormous fraction of the world is invisible to your model. A user asks "what is this?" and points at a photo of a circuit board. A doctor asks "what do you see in this chest X-ray?" A child asks "why is the dog wearing a hat?" None of these conversations are reachable by a text-only system. Building a model that reads pixels is not a frontier capability — it is the bare minimum required to be useful in most real situations.

There is a second reason to care, and it is more subtle. The bridge from vision to language teaches you something general about how to bolt one model onto another. The same trick works for audio, for video, for sensor streams from a robot. Each modality gets its own encoder. Each encoder feeds a projection. Each projection lands in the decoder's embedding space. Once you have written one of these by hand, the others follow without conceptual surprise. This chapter is the practice case for an entire family of architectures you will see again.

A third reason is the failure mode you will see in BREAK IT. The projection layer is small. The encoder and decoder are large. A reasonable engineer might assume the projection is the easy part. Surely the hard work is the encoder and decoder, and the projection is just plumbing. That assumption is wrong, and watching it fail in this chapter is one of the cleanest demonstrations in the book that the small bridge between two big networks is where the actual multimodal capability lives.

Here is my strongest opinion in this chapter, said plainly: the projection IS the multimodal capability. Not "part of" it, not "an important component of" it. It IS it. Everything else is two pretrained networks doing what they were already doing, in their own embedding spaces, blind to each other. The projection is the only thing in the entire pipeline that has to learn the cross-modal correspondence. If it does not learn, nothing else can compensate.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/29_multimodal-a-tiny-vision-language-model/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/29_multimodal-a-tiny-vision-language-model/build.py --full

# The BREAK IT experiment:
python projects/29_multimodal-a-tiny-vision-language-model/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 29 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
