# Project 21: Fine-Tuning And Instruction Tuning

## Hook

Why does a pretrained model that can explain recursion, write a poem, and finish your code still fail the most basic request pattern in chat: you ask a question, and it keeps roleplaying the rest of your prompt instead of answering it? That feels like a bug. It is not a bug. It is the whole point of this chapter.

Pretraining teaches a model to continue text. Fine-tuning teaches it which text shape you care about. Instruction tuning teaches it that `"User: ..."` should be followed by `"Assistant: ..."` instead of more user text. The weird part is that this change in behavior can come from a tiny amount of additional training compared with pretraining, and it can also go wrong in ways that are almost embarrassingly easy to trigger.

I have a stronger reaction to this chapter than most. The long fine-tuning sweeps I have been running for years live inside this exact problem. Four model families, 410M to 6.9B parameters, thousands of training runs. After the first couple hundred, I stopped being surprised that the model improved. I started being surprised by how often it improved in the wrong direction. This chapter is the version of that experience I would have wanted before I started.

## The Concept

Start with a person who has read an entire library but has never worked a help desk. If you hand them a support ticket and say, "Answer the customer directly," nothing in "read a library" guarantees the format you want. That is a pretrained model.

Pretraining is broad reading. It teaches the model world knowledge, syntax, code patterns, and statistical habits of text. It does not teach a stable job description. Fine-tuning is job training. You take the same person and say: when the input looks like this, respond like that. When the user asks a question, produce an answer. When the prompt asks for JSON, produce JSON. If the format has `user` and `assistant` turns, continue with the assistant turn only.

The first time I watched this transition happen on a checkpoint I had trained myself, it was almost unsettling. The base model produced fluent nonsense. The same weights, after a few thousand SFT steps on conversation data, produced answers. Nothing was added to the model in any meaningful sense. The mass had just been pushed around.

Instruction tuning is a specific kind of fine-tuning. You are not just saying "learn more text from this domain." You are saying "learn the social contract of a request-response interface." Fine-tuning does not give the model a new brain. It nudges the one it has. The same parameters that stored "Python looks like this" and "medical notes look like this" now get shifted toward "when you see an instruction, answer it in this style." That is why fine-tuning is so effective: you are not building knowledge from scratch, you are steering existing knowledge into a more useful interface. Fine-tuning is also fragile. Too little data invites memorization. Sequential tuning invites forgetting. Bad data teaches bad behavior.

LoRA, short for Low-Rank Adaptation, exists because full fine-tuning is expensive. Plain-English version first: instead of sanding down and repainting the whole car, you bolt on a small adjustment kit that changes how the steering behaves. The original car stays there, your add-on learns the task, and you save memory because you train the add-on, not the whole machine.

The mental model that finally worked for me, after a few months of adapter sweeps, was thinking of LoRA adapters as removable lenses you screw onto the same camera body. The body is the base model. The lenses change what the camera does without modifying the sensor. You can swap them. You can stack them. And you can lose them, which matters more than I expected.

That "low-rank" phrase needs demystifying. A weight matrix is just a giant table of numbers that maps one list of numbers to another. A full update says: every number in that table may change. LoRA says: keep that giant table frozen and learn two much smaller tables whose product acts like a compact correction. Not every possible change, just the changes you can express through a small bottleneck. It is restrictive, but task-specific behavior often lives in a much smaller subspace than the full model, so a few well-chosen directions can be enough.

Now the failure mode everyone eventually meets: catastrophic forgetting. Imagine teaching that same help-desk worker to answer legal questions for a week, then medical questions for a week, but never letting them practice legal questions again. By the end, their phrasing, habits, and retrieval cues drift toward medicine. They did not lose all legal knowledge, but their behavior on legal prompts gets worse. The second training stage overwrote part of the behavioral map built by the first. That is not an edge case. It is the default if you fine-tune sequentially without protection.

I learned this the embarrassing way. On one edge deployment for community health workers, I fine-tuned a small checkpoint on Tamil medical triage prompts, then resumed training on English protocol text because I thought I was being efficient. The Tamil output collapsed. The model still spoke Tamil. It just stopped sounding like a clinician.

One more thing matters more than most introductions admit: what tokens get counted in the loss. Loss is the number that says how wrong the model is. During instruction tuning, you usually do **not** want to punish the model for the user's prompt tokens. You want to punish it only for the assistant's reply tokens. The reason is that the prompt is given to the model; the model is not supposed to "predict the user" as the main task. It is supposed to predict the assistant answer conditioned on the user prompt.

nanochat makes this explicit. In its SFT pipeline, the conversation is rendered into tokens plus a mask, and only assistant completion positions count toward the training target. That one detail is a big deal: it turns "model conversations" into "learn to answer conversations." Without that mask, the model spends capacity learning to copy the prompt pattern more than it should. With the mask, you are teaching the role boundary itself.

## Why It Matters

Without fine-tuning, your model has knowledge without behavior. It can continue a paragraph about APIs, continue a stack trace, or continue `"User: How do I sort a list in Python?\nAssistant:"` in a way that sometimes looks like an answer — but it has no strong obligation to answer. It is guessing the next token, not honoring a contract. That difference sounds philosophical until you test it.

Prompt a base model:

```text
User: Explain closures in JavaScript.
Assistant:
```

A base model may continue with another `User:` turn, produce a partial answer that drifts into unrelated continuation, imitate forum text, or write a transcript that looks plausible without staying on task. Prompt an instruction-tuned model with the same text and it tends to answer directly. That is the product difference users feel.

The first time I demoed this side-by-side to a non-engineer, they did not understand what they were looking at. The base model continued the prompt. They said, "but it sort of answered." I had to explain that the model had no obligation to answer. It happened to drift in the right direction sometimes. That distinction is invisible until you watch it fail on a hundred prompts.

Fine-tuning also lets you specialize. You can take a general model and make it better at:

- customer support
- medical summarization
- code review
- legal clause extraction
- tool calling
- structured outputs

But there is no free lunch here. Full fine-tuning updates every parameter, which usually gives the model the most room to adapt but costs the most memory and is the easiest way to accidentally damage old behavior. LoRA updates far fewer parameters, making training cheaper and deployment cleaner; you can keep one base model and swap tiny adapters for different tasks. But LoRA is a trade: cheaper adaptation in exchange for less freedom to reshape the model.

My honest opinion, after enough adapter runs to be slightly tired of them: most teams pick LoRA for the wrong reason. They pick it because it fits on their GPU. The correct reason to pick LoRA is that it lets you keep the base model preserved. The cost savings are real but secondary. The structural property that you can ship multiple adapters against the same frozen base is what changes how you build products.

Then there is forgetting. If you plan to build systems that evolve over time, forgetting is not a theory question. It is a product question. You fine-tune on support tickets in January, fine-tune on policy Q&A in February, and discover in March that your support bot now answers support questions in policy language and misses edge cases it handled before. That is not the model becoming worse in general. That is the model becoming more aligned to the newest pressure you applied.

This chapter matters because post-training is where a text model turns into a product, and also where many teams quietly damage their models while thinking they are improving them.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/21_fine-tuning-and-instruction-tuning/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/21_fine-tuning-and-instruction-tuning/build.py --full

# The BREAK IT experiment:
python projects/21_fine-tuning-and-instruction-tuning/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 21 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
