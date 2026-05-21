# Project 25: Test-Time Reasoning (CoT, Self-Consistency, Best-of-N)

## Hook

Your fine-tuned model from **Project 21: Fine-Tuning and Instruction Tuning** answers "what is 2 + 2?" without hesitation. Ask it a four-step grade-school word problem and it confidently produces the wrong number. Nothing about the model has changed. The same weights that nailed the easy question whiffed on the harder one. The math the model needs is sitting inside those weights. It just is not getting deployed. What compute can you spend at inference time, after training is over and the weights are frozen, to fix this?

The first time this happened to me on an ACAR-style routing experiment, I assumed the weights were broken. They were not. The model was being asked to do four steps of arithmetic with no place to write down the intermediates, and it was guessing the answer in one forward pass. That was the lesson. The chapter is about the cheapest fixes available.

## The Concept

The model has the capability. It does not have the deliberation. A student who knows how to add can still mess up a long arithmetic problem if they try to do every step in their head and only write down the final number. The same student, given a piece of scratch paper and told "show your work," makes fewer mistakes. The information needed to be correct was always in their head. The scratch paper changed what they could keep track of.

A language model has the same problem. When it generates a single answer token in response to a hard problem, it is doing every intermediate calculation in one forward pass, with no way to write anything down. **Chain-of-thought**, often shortened to **CoT**, is the scratch paper. You prompt the model with something like "Let's think step by step," and the model produces a sequence of intermediate reasoning tokens before giving its final answer. The final answer is now generated with all those intermediate tokens already in its context, so it has a much easier job.

That is one move. There are three more, and each one buys accuracy by spending more inference compute.

**Self-consistency** is the second move. Generate not one chain of reasoning, but eight. Sample them at a temperature high enough that the chains genuinely differ — temperature 0.7 is the usual setting. Read off the final answer from each chain. Then take a **majority vote**: whichever final answer appears most often across the eight chains is the answer you keep. The intuition is that there are many wrong ways to reason about a hard problem, but most of the right ways tend to converge on the same number. If five out of eight chains say "47" and the other three say three different wrong things, "47" is your best guess.

**Best-of-N**, written **BoN**, is the third move. Sample N candidate answers — full chains plus final answers — and pick the best one using a **verifier**. The verifier is a separate model trained to score answers, often called a **reward model** or **outcome reward model** (**ORM**) when it scores final outcomes. An ORM takes a question and a candidate solution and produces a number: higher is better. You take the candidate with the highest ORM score and return that one. Majority vote uses agreement to pick. BoN uses an external judge.

**Process reward models** (**PRMs**) are the fourth move. An ORM scores the final answer. A PRM scores each individual step in the reasoning chain: was step one correct, was step two correct, and so on. With a PRM you can do something more surgical than picking among finished candidates. At each reasoning step, you sample K continuations, score each one with the PRM, keep the best step, and continue. The search is now step-level rather than candidate-level. It is more compute, but the verifier can reject bad reasoning before it derails the whole chain.

PRMs are also where my opinion on this whole space gets sharpest. I think PRM training is one of the most fragile pipelines in the test-time playbook, and the literature understates how easy it is to ship a PRM that grades the wrong feature. The BREAK IT later in the chapter exists because I have seen this failure mode in the wild, on fusion candidates where a verifier was rewarding chain length and we mistook it for rewarding correctness for almost a full day.

A useful analogy to hold all four together: a student doing a math problem. The student who shows their work catches their own arithmetic mistakes; that is CoT. Three students working in parallel and majority-voting on the final answer do better than any one of them; that is self-consistency. A teacher who reads each student's final answer and picks the best one does better still; that is BoN with an ORM. A teacher who reads each step as the student writes it and tells them when they have gone wrong does the best of all; that is a PRM doing step-level search.

There is a search-tree version of the last move that is worth naming separately. **MCTS** stands for **Monte Carlo Tree Search**, and it generalizes step-level BoN into a real tree search. From the current partial reasoning, expand several candidate next-steps, score them with the PRM, recurse into the most promising one, occasionally back up and try a different branch when scores stop improving. The math is more involved, but the structure is the same: PRM-guided search through the space of possible reasoning chains. A **rollout** in this context is one full path from the current node down to a final answer. MCTS does many rollouts and uses their outcomes to decide which branches to develop further.

All five methods (direct answering, CoT, self-consistency, BoN, step-level BoN with a PRM, and MCTS) trade inference compute for accuracy. Direct answering is the cheapest and least accurate. MCTS is the most expensive and, when the PRM is good, the most accurate. There is no free improvement here. You are paying for accuracy in GPU-seconds at inference time, the same way you paid for it in GPU-hours at training time.

I think of this as the budget version of adaptive complexity routing. Multi-model routing work I have done across single, two, and three-model ensembles was solving the same problem at a higher level: when does it pay to spend more compute on a question? Test-time reasoning is the same question one rung down, inside a single model. The answer is also the same: sometimes a lot, sometimes nothing, and you cannot tell ahead of time without measuring.

## Why It Matters

Without test-time reasoning, your model is leaving capability on the table. The training run already happened. The weights already encode whatever they encode. You can either ask the model to one-shot every hard problem and live with the failure rate, or you can spend a little more compute at inference and recover accuracy you already paid for during training.

The numbers in the literature are large enough to take seriously. On GSM8K, a grade-school math benchmark, naive prompting of a mid-sized model gets you maybe 30 percent accuracy. CoT prompting on the same model jumps to 50 percent or higher with no weight changes. Self-consistency on top of CoT, with eight samples, pushes that another five to ten points. BoN with a trained verifier can push it again. The o1 family of models from OpenAI, and the public discussion around them, leans heavily on this trick. The headline "reasoning" capability is in large part a story about spending more compute per query at inference time, using methods that are direct descendants of the ones in this chapter.

There is a second reason this chapter sits where it does. The previous chapters in this book taught you to spend training compute carefully. **Project 23: Reward Models and RLHF** taught you to train a reward model on preference data. **Project 24: DPO and Preference Optimization** taught you a cheaper alternative that bakes preferences into the policy directly. Test-time reasoning re-opens the question those chapters seemed to close. Maybe you do not need to bake every behavior into the weights. Maybe some behaviors are cheaper to get by spending compute at inference time, with a smaller policy and a smaller reward model. That tradeoff is one of the live research questions of the period this book is being written in.

A third reason, less direct but worth saying. Test-time methods make capability cheaper to study. You do not need to retrain anything to see whether the model "knows" how to do a task. Crank up the inference compute, watch what happens to accuracy, and you have a much clearer answer than any single-shot eval. **Project 22: Evaluation Methodology** taught you to be careful with benchmarks; test-time reasoning is one of the reasons that care matters.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/25_test-time-reasoning-cot-self-consistency-best-of-n/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/25_test-time-reasoning-cot-self-consistency-best-of-n/build.py --full

# The BREAK IT experiment:
python projects/25_test-time-reasoning-cot-self-consistency-best-of-n/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 25 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
