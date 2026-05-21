# Project 22: Evaluation Methodology

## Hook

**Project 9: Pretraining on the Real Web**'s `val_bpb` told you the model was learning. The number went down for a week straight, the curves looked clean, and the validation loss matched what the literature reported for a model that size. Then you scored the same model on MMLU and got 12 percent. Barely above the random-guessing floor for four-choice questions. Then you scored it on a benchmark you wrote yesterday and got 78 percent. Same model. Same weights. Same checkpoint. Three numbers that disagree by a factor of six. How can all three be right?

This is the chapter I most wanted to write and most dreaded writing. Multi-model routing work I have done across ensembles touched roughly 1,500 tasks, and the headline accuracy number is uninterpretable without three pages of methodology. Most of the noise in published model comparisons today is people quoting headline numbers without those three pages. My strong opinion is that benchmark numbers without methodology are not numbers. They are press releases with decimals attached.

## The Concept

Imagine a student preparing for a final exam. You have three ways to measure whether the student learned the material. You can ask them to recite the textbook from memory, which tests recall. You can give them a multiple-choice quiz, which tests pattern matching against four answer shapes. You can hand them an open-ended problem and grade their reasoning, which tests something closer to understanding. Each method gives you a different number, and each number measures something different. None of them is "the truth." All three together start to look like one.

Now add a twist. Suppose the multiple-choice quiz is identical to the practice quiz the student took last week, and they wrote down every answer. The score will look brilliant, but the student has not learned anything new — the test is broken in a hidden way. This is the situation every honest evaluator of large language models lives in.

There are four families of evaluation that matter, and each measures a different slice of the model.

The first is **perplexity**, or its byte-normalized cousin, **bits per byte** (often abbreviated bpb). Perplexity asks a narrow question: how surprised is the model by a piece of natural text it has never seen? You feed the model a held-out sentence, you compute the probability the model would have assigned to each token in that sentence, and you take the geometric mean. Low perplexity means the model finds the text predictable. Bits per byte normalizes by the actual byte count of the text, which lets you compare models with different tokenizers. It is the unit **Project 9: Pretraining on the Real Web** already used to compare runs. Perplexity is the cleanest signal you can get, but it only measures distribution fit on natural text. It does not measure whether the model can follow instructions, answer questions, write code, or do anything you would actually use a language model for.

The second is **multiple-choice scoring**, where the model picks A, B, C, or D. The most famous benchmark in this family is MMLU, Massive Multitask Language Understanding, which gives the model 57 different subjects, from high school physics to professional law, and asks four-choice questions. The way you score it matters a great deal. The standard approach computes the **logprob** (the log of the probability) of each answer token under the model's distribution, conditioned on the question and the answer choices. You pick the option with the highest logprob and check whether it matches the labeled answer. Multiple-choice scoring is fast, repeatable, and brittle. The model might pick the right letter for the wrong reason, or the wrong letter for a reason the format does not let it express.

The third is **open-ended generation evaluation**. You give the model a prompt, you let it generate a response, and you check whether the response is correct. The check can be exact-match (does the generated text match the reference exactly?), or substring match (does the generated text contain the reference answer somewhere?), or it can use a more permissive comparison. Generation eval measures something closer to real usage, but the score depends on the prompt format, the sampling temperature, the maximum length, and a dozen other choices that the multiple-choice version hides.

The fourth is **LLM-as-judge**, where a stronger model reads the candidate model's output and assigns a score on a rubric. You write a prompt that says "Rate this answer from 1 to 5 on correctness and clarity," you send the candidate's output to GPT-4 or Claude, and you collect the judge's score. LLM-as-judge is the closest you can get to measuring usefulness at scale without paying human labelers, but the judge has biases. It prefers longer responses. It prefers the first response when shown a pair. It tends to agree with confident-sounding text even when the text is wrong.

When I first ran a Claude-as-judge pipeline on ACAR outputs, the agreement rate with my own ratings was around 71%. That sounds reasonable until you notice the disagreements clustered. The judge confidently endorsed verbose wrong answers I had marked down. I rewrote the rubric three times before the bias stopped showing up in the failure modes I cared about.

All four methods are valid. None of them measure the same thing. A model can score 4 bpb on web text, 12 percent on MMLU, 80 percent on a custom benchmark you wrote, and 3.4 out of 5 from a Claude judge, all at the same time. The numbers are not contradictory. They are answering different questions.

There is one more concept worth naming before any code appears: **contamination**. Contamination happens when the evaluation set leaks into the training data. Maybe the MMLU questions were posted on a forum that got scraped into the web crawl. Maybe a paper reproduced the GSM8K problems in its appendix and that paper ended up in your training corpus. Maybe someone bundled a public benchmark into a documentation repository and your data pipeline included it. The leak does not need to be deliberate. It just needs to happen.

When the eval set leaks, every evaluation method inflates. Perplexity drops on the leaked passages because the model has memorized them. Multiple-choice scores rise because the model has seen the question and the answer. Generation evaluations spike because the model can reproduce the reference output verbatim. LLM-as-judge ratings climb because the candidate's output suddenly looks correct, fluent, and confident, for the wrong reason. The model has not learned anything. It has memorized the test. The test is not broken, but the score is a lie.

The analogy that nails this is the cake-baking analogy adjusted slightly. A student scores 100 percent on a final exam that turns out to be a copy of last week's homework. The student has not learned the material; the score reports memorization, not understanding. The test is not broken. The score is. A score divorced from how the test was constructed and whether it was held out is, by itself, uninterpretable.

The mental image that finally made contamination feel concrete to me was thinking of the eval set as a sealed envelope. The training data was supposed to never see the inside of that envelope. The contamination check is the dusting for fingerprints. Most envelopes are clean. Some are covered.

This chapter teaches the four methods, shows how each one fails, and builds the contamination detector that catches the silent failure underneath them all.

## Why It Matters

Without proper evaluation methodology, the field of language modeling becomes a casino. Researchers publish benchmark numbers. Practitioners read those numbers and pick models. Buyers read marketing copy that quotes the same numbers and sign contracts. Every step of that chain assumes the numbers mean something specific. When the numbers are produced by a contaminated benchmark, or by a scoring method nobody documented, or by a prompt template that happens to favor one model, the entire chain breaks silently. The decisions still happen. They are just based on noise.

There is a simpler reason this chapter sits where it does. After **Project 21: Fine-Tuning and Instruction Tuning** and before **Project 23: Reward Models and RLHF**, you are about to spend real compute on shaping model behavior. Every shaping technique needs a way to verify that it worked. If your eval inflates whenever the training data overlaps the test set, and it does, every time, you will train against the wrong signal and ship a model that scores well on paper while behaving worse in practice. You will not know this happened. You will think the training worked.

Years of running a multi-state SaaS platform taught me one thing that applies here, even though that work had nothing to do with language models: measurement at scale is mostly about discipline, not about the metric. A 99.99% uptime number means very little if you cannot show me what counted as downtime, who was watching, and which incidents got rolled into "scheduled maintenance." Benchmark numbers fail the same way. The score is the surface. The methodology underneath is where the truth lives or hides.

There is also a longer-running reason that becomes obvious once you have read three or four evaluation papers. The published benchmark numbers from major labs are nearly impossible to compare directly. One lab reports MMLU using 5-shot answer-token scoring. Another lab reports MMLU using zero-shot full-completion scoring with a different prompt template. A third lab fine-tunes on MMLU-adjacent data and reports the result anyway. The headline numbers look like they should be comparable. They are not. The skill this chapter builds is the ability to read a benchmark number and ask the three questions that decide whether it is meaningful: how was it scored, how was the prompt constructed, and was the eval held out from training. Without those answers, the number is meaningless. With them, you can interpret almost any published score.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/22_evaluation-methodology/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/22_evaluation-methodology/build.py --full

# The BREAK IT experiment:
python projects/22_evaluation-methodology/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 22 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
