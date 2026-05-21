# Project 28: Retrieval-Augmented Generation

## Hook

Your **Project 21: Fine-Tuning and Instruction Tuning** model knows what was in its training data, and only what was in its training data. Ask it about a paper that came out last month, a customer's billing history, or an internal wiki page, and it makes something up that sounds right. Retraining the model every time a new document arrives is absurd, both in cost and in latency. So how do you teach a model what it does not know, without touching its weights? That is the question this chapter answers, and the answer is a system, not a model.

## The Concept

The trick is to separate two jobs that the model has been doing badly because it was forced to do them together. One job is reasoning: given some text, write a coherent answer. The other job is recall: pull up the specific facts the answer should be based on. A pretrained model is excellent at the first job and unreliable at the second. So you stop asking it to do the second job. You give it the facts at query time and let it focus on writing.

Picture a student writing an essay in a library. The student is good at sentences, structure, and argument. The student cannot remember every book in the library, and you would not want them to. So a librarian sits next to them. When the student needs a fact about, say, the Treaty of Westphalia, the librarian walks over to the right shelf, pulls down three books, and slides them across the table. The student reads the relevant pages and writes the paragraph. The student never had to memorize the library. The librarian never had to write the essay. Each role is good at a different job, and the system as a whole produces a better essay than either could alone.

That analogy is the one I keep coming back to whenever I explain RAG to someone who has only worked with LLMs as a black box. The instinct most people have is to make the student smarter. The actual move is to hire the librarian.

Retrieval-Augmented Generation, or RAG, is the same arrangement built out of code. There are three components and they correspond exactly to the analogy. The first is an **embedding model**, a small neural network that converts a piece of text into a list of numbers, called a vector. Texts with similar meaning produce vectors that point in similar directions. The embedding model is the librarian's sense of which books are about which topics. The second is a **vector index**, a data structure that stores millions of these vectors and can quickly find the ones closest to a query vector. That is the library itself, organized so the librarian does not have to scan every shelf. The third is a **generator**, which is your existing LLM from earlier chapters. That is the student.

The flow at query time is short. A user asks a question. The embedding model converts the question into a query vector. The vector index returns the top few documents whose vectors are nearest to that query vector. The system stitches those documents into the prompt, with a small header that says "here is some context, use it to answer." The generator then produces an answer conditioned on that context. The model weights are not changed. The index is what holds the knowledge, and the index is something you can update in seconds.

A small piece of vocabulary, since we will use it for the rest of the chapter. The embedding model that produces vectors for the documents and the query independently is called a **bi-encoder**, because it has two separate encoding paths even though they share weights. The model that takes a query and a document together and scores how well they match is called a **cross-encoder**. Bi-encoders are fast and approximate. Cross-encoders are slow and accurate. We will use the bi-encoder to find a hundred plausible candidates, and the cross-encoder to pick the best three from those hundred. That two-stage pattern is how serious systems are built.

The deep point underneath all of this is that knowledge and reasoning are different kinds of work, and combining them inside one set of frozen weights is a poor design. The model's parameters store regularities of language, plus whatever facts happened to be in the training corpus, fused together in a way you cannot edit. The index stores facts in a form you can add to, replace, or delete this afternoon. When the facts change, you update the index, not the model. The student keeps writing essays. The library keeps growing.

Here is the take that I will defend through the rest of the chapter. RAG is a structured-prompt problem dressed up as a retrieval problem. The hard part is almost never the vector math. The hard part is the contract between the retriever and the generator: what gets retrieved, how it gets formatted into the prompt, and what the generator does when the retrieval is wrong. I have shipped retrieval on a multi-state SaaS platform handling hundreds of thousands of customer records across dozens of sites. Most of the production failures lived in the prompt-and-contract layer. Almost none of them lived in the index.

## Why It Matters

Without retrieval, your model can only answer questions about its training data, frozen at the date the training finished. Anything newer is invisible. Anything proprietary is invisible. Anything domain-specific that the open web did not contain is invisible. You can ask the model anyway, and it will produce a confident answer, which is the worst possible failure mode because the answer is wrong but reads as if it is right. This pattern has a name now in the literature, **hallucination**, and it is the single largest reason that production deployments of LLMs go sideways. RAG is the most effective practical fix.

There is also a cost argument. Fine-tuning a model on a new corpus costs GPU-days and engineering time, and the result is brittle: when next month's documents arrive, you fine-tune again. Building an index over the same corpus costs minutes. Adding a new document is one embedding call and one index update. When a document is wrong or out of date, you delete it from the index and the next query no longer sees it. None of this requires touching model weights.

There is one more reason this chapter matters and it is a forward-looking one. The next generation of agent systems, the ones in **Project 26: Tool Use and Function Calling**, treat retrieval as one tool among many. The agent decides when to retrieve, what to retrieve, and how to refine the query if the first retrieval did not return useful documents. None of that works if the retrieval primitive is broken. Get this chapter right, and the agentic systems that come later inherit a sound foundation. Get it wrong, and every agent built on top inherits the same upstream failure.

On multi-model routing work across several frontier models, the routing logic sat right on top of a retrieval primitive that fed evidence into the model that was about to answer. When the retrieval was bad, the routing accuracy dropped in ways that I initially attributed to the router. It was not the router. It was the upstream facts.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/28_retrieval-augmented-generation/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/28_retrieval-augmented-generation/build.py --full

# The BREAK IT experiment:
python projects/28_retrieval-augmented-generation/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 28 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
