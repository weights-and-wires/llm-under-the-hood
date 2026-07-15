# Project 17: Production Serving: Continuous Batching and PagedAttention

## Hook

**Project 13: Fast Inference: The KV Cache** made one request fast. It cached the keys and values from earlier tokens so the model never recomputed them, and it turned a quadratic-cost chat into a linear one.

Now imagine a hundred users hit the same model at the same second, each with a different prompt length, each wanting a different number of tokens back. One typed two words. Another pasted a thousand-line document. A third asked a yes-or-no question that finishes in three tokens. A fourth wants a five-thousand-token story.

Your single-request KV cache code, run a hundred times in parallel, falls apart almost immediately. It pads everything to the longest prompt, allocates a huge contiguous buffer per request, makes the short queries wait behind the long ones, and the GPU spends most of its time twiddling its thumbs on padding. This chapter is about the two ideas that fix that.

A quick note before we start: this is a proxy lab on a single GPU. You will simulate a hundred concurrent users locally (a hundred independent request states managed by one Python scheduler on one box) because most readers do not have a production serving cluster and do not need one to learn the mechanics. The lessons transfer up to the cluster scale almost unchanged; the code you write on one GPU is structurally the same code vLLM and TensorRT-LLM run on dozens of GPUs.

I came to this chapter the long way around. Years before I touched an LLM, I was at Cognizant on healthcare claims systems where zero-downtime deployments and steady tail latency were the only metrics that mattered. The shapes of the problems repeat. Variable arrivals, variable work per request, a finite resource pool, and a scheduler whose job is to keep utilization high without ever leaving a request stuck. The vocabulary in LLM serving is new. The problem is not.

## The Concept

There are two separate problems in the way of efficient serving, and they get mixed up constantly. The first is about time. How the scheduler decides which request occupies which slot of the batched forward pass at each step. The second is about space. How the KV cache for many concurrent requests shares the GPU's finite memory without falling apart into useless fragments.

Continuous batching solves the time problem. PagedAttention solves the space problem. They are usually shipped together because they are useless apart: a great scheduler that runs out of memory after twelve requests is no better than a great memory allocator with a scheduler that wastes 90% of its compute on padding. The chapter walks through them one at a time, then puts them together.

### Static batching, the thing you would build first

A neural network is happiest when you feed it many examples at once. The GPU can do the same operation across a thousand vectors in roughly the time it would take for one. So your first instinct when serving an LLM to many users is the right one: collect a batch of requests, run them through the model together, and decode one token from each on every forward pass. That is **static batching**. It is the way every introduction to deep learning serving starts. It is also a trap.

The trap is that LLM serving is not like classifying images. In image classification, every example is the same size and the work is the same per example. In LLM serving, requests are the wrong shape for batching in two ways. First, prompts have wildly different lengths (a tweet versus a contract). Second, responses end at unpredictable times. One request hits its end-of-sequence token after three decoded tokens, another after five hundred. Static batching tries to force these mismatched workloads into one big rectangular tensor, and it fails twice over.

Picture a school bus that stops at five houses to pick up kids and only drives away when everyone has finished getting on. Now imagine that same bus drops everyone off at the same final stop, and waits at every intermediate stop until the slowest passenger says they are ready to get off. Static batching is that bus.

Every request waits at the start until enough requests have been queued and padded to the same length. Then every request rides together. Then every request waits at the end until the longest response is done. Your three-token "yes" sits there for five hundred decode steps so a thousand-token essay can finish next to it. This is **head-of-line blocking**: short jobs queueing behind long ones. It is the single biggest source of wasted compute in a naive LLM serving stack.

### Continuous batching, the thing you actually want

The fix is to stop treating the batch as a fixed group that enters together and leaves together. Instead, the batch is a turnstile. At every decoding step (every single forward pass) the scheduler looks at every active request, retires the ones that just produced an end-of-sequence token, frees their slots, and admits new requests from the queue into those slots. The batch is the set of requests being decoded right now, and the membership changes at every step.

This is **continuous batching**. The name is slightly unfortunate because what is continuous is not the batch but the eligibility. Every request, every step, is eligible to enter or leave. The original paper (Yu et al., "Orca," 2022) called it iteration-level scheduling, which is clearer if less catchy. The unit of scheduling is one forward pass, not one full request.

Picture a turnstile at a subway station. People arrive at different times. Each passes through when it is their turn and leaves when they reach their stop. Nobody waits for anyone else. The turnstile clicks at a constant rate. The bus wastes time waiting for the slowest passenger; the turnstile never waits at all.

One subtlety worth flagging early. A new request joining the batch mid-stream is not yet decoding. It still needs its prompt processed. The first forward pass for a new request is a **prefill** step (running the model over the prompt to produce the initial KV cache), and the subsequent passes are **decode** steps (generating one token at a time using that cache). Most production servers either interleave prefill and decode in the same batch using clever masking, or they run a small dedicated prefill pass before merging into the decode batch. We will build the simpler version first and discuss the trickier one in the homework.

### PagedAttention, the thing you build for memory

Now turn to the second problem. Every active request has its own KV cache, the sequence of keys and values for every token decoded so far. In **Project 13: Fast Inference: The KV Cache** you allocated one big contiguous tensor per request, sized to the maximum sequence length you might ever generate. That works for one request. For a hundred requests it does not.

Why not? The maximum length is much larger than the average length. If the maximum is 4096 tokens but the average response is 200 tokens, you have pre-allocated room for 4096 tokens of cache per request and you will use, on average, 5% of it. The other 95% is reserved memory you cannot give to anyone else.

Run a hundred requests and you have reserved enough cache for 400,000 tokens but you are actually storing about 20,000. That gap is **fragmentation**: memory that is technically allocated but actually wasted. It limits how many requests you can run concurrently on a GPU before you hit out-of-memory errors.

Low-level kernel work on Apple Neural Engine kept pushing me toward this same problem from a different angle. IOSurface buffer constraints meant you could not pretend the buffer pool was infinite, and you could not pretend every request would use the maximum it might use. Edge inference scheduling forced exactly this question: how do you give every active request enough cache to make progress without reserving so much that no new request can arrive?

The fix comes from operating systems. The OS solved this exact problem decades ago for process memory. The idea is **paging**: instead of giving each process one large contiguous slab of memory, give each process access to many small fixed-size pages, and let the OS map those pages on demand. Each process thinks it has a linear address space; under the hood, a page table translates every virtual page to a physical page wherever it happens to live in RAM. The pages do not need to be contiguous. A process grows its address space one page at a time, and the OS allocates pages from a free list and reclaims them when the process exits.

**PagedAttention** is paging applied to the KV cache. The KV cache for each sequence is broken into fixed-size pages, typically 16 tokens per page. Each sequence has a **page table** that maps its logical token positions (token 0, 1, 2, ...) to physical pages somewhere in a big pool of pre-allocated GPU memory. A **free list** tracks which physical pages are unused. When a sequence needs more cache, the allocator grabs a page from the free list and adds it to the page table. When a sequence finishes, all its pages return to the free list. The maximum sequence length is no longer a per-request reservation; it is just an upper bound on how many pages a single sequence might use if it runs to the end.

Fragmentation collapses. With pages of 16 tokens, the worst case is that the last page of a sequence is mostly empty (say, 4 tokens used out of 16). That is at most 12 tokens of waste per sequence, regardless of length. Across a hundred sequences, 1,200 tokens of waste total. Compare to the contiguous case, where each sequence might waste thousands, and the difference is two orders of magnitude.

The picture: contiguous KV allocation is like a parking lot where every car gets a 50-foot space whether they brought a motorcycle or a truck. Paged KV allocation gives every car exactly as many 8-foot stalls as it needs, picking from any open stalls in the lot. A motorcycle takes one stall. A truck takes six. The lot fits more vehicles.

### The two ideas together

Continuous batching gives the GPU a steady stream of work without head-of-line blocking. PagedAttention gives the memory allocator a way to fit many concurrent requests without burning gigabytes on padding. Either alone is helpful; both together is what makes a single GPU serve dozens or hundreds of concurrent users. This is the recipe behind vLLM (Kwon et al., 2023), the project that made these ideas standard practice.

Two more terms before we build. **Admission control** is the policy that decides when to admit new requests from the queue. You cannot admit more requests than you have pages for, or you will run out mid-generation. **Preemption** decides what to do when you have admitted too many requests and you are about to run out anyway: you have to evict somebody and run their request later. We will not implement full preemption, but we will demonstrate what happens without admission control, which is the most instructive failure mode in the chapter.

This is the part I want to be opinionated about: most teams skip admission control until production breaks, and then they bolt it on under pressure. Build it first. The system that has admission control from day one and never uses it costs you very little. The system that does not have it and needs it at 2 AM costs you everything.

## Why It Matters

The cost of LLM inference at any scale is dominated by GPU time. A single H100 costs roughly the same per hour whether it is running at 5% utilization or 95%. The job of the serving stack is to push utilization as close to 100% as possible without crashing. Continuous batching plus PagedAttention is the difference between a stack that handles 10 concurrent users per GPU and one that handles 100. That is a tenfold difference in cost per token, every minute of every day, for the life of the deployment.

In a past role on a multi-state SaaS modernization, the uptime target was 99.99%. That is roughly 52 minutes of allowed downtime per year. The arithmetic of that target shaped every decision about queues, retries, capacity headroom, and shedding. LLM serving has the same arithmetic, only the cost units are GPUs instead of database connections.

If you operate any product that calls an LLM, this matters even if you do not host the model yourself. Every API provider runs some version of this machinery. Their pricing, their latency, their rate limits, their behavior under load are all shaped by these two algorithms. Understanding what is happening on their side helps you reason about why your costs spiked when traffic doubled, why your p99 latency cratered when one tenant submitted a flood of long prompts, and why "context caching" features started showing up in API price sheets last year.

The skill also transfers downward. These algorithms exist because LLM inference has unusual workload characteristics: variable arrival times, variable input sizes, variable output sizes, and stateful per-request memory. Any system with those characteristics can borrow these patterns. Vector search with per-query reranking, large-model finetuning over many tenants, batch transcription with wildly different audio lengths. All benefit from a continuous-batching scheduler and a paged allocator. Once you have built it once, you start seeing the pattern in places that were never about LLMs at all.


Several future chapters depend on this one. **Project 14: Speculative Decoding** speeds up each request by guessing several tokens ahead with a small model and verifying them with the big one. Speculative decoding interacts with the scheduler: accepted tokens advance multiple positions, rejected drafts return to the queue. You cannot wire that in cleanly without first having a scheduler. **Project 15: Grouped Query Attention** changes the shape of the KV cache and therefore the page sizes used by the paged allocator. Both land more smoothly once continuous batching and paging are familiar.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/17_production-serving-continuous-batching-and-pagedattention/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/17_production-serving-continuous-batching-and-pagedattention/build.py --full

# The BREAK IT experiment:
python projects/17_production-serving-continuous-batching-and-pagedattention/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Hardware validation

The 2026 update discusses fp8 KV-cache quantization and speculative decoding as two serving levers, and how they compose. Those claims were run end-to-end on an NVIDIA GB10 (DGX Spark) serving Qwen3-8B under vLLM — charts, raw benchmark JSON, and a reproduce script are in [`validation_fp8_kv_gb10/`](validation_fp8_kv_gb10). Headline: fp8 KV holds ~2.07× the token budget of bf16, and a causal n-gram drafter composes with it for ~1.9× throughput at 71% draft acceptance.

## Read in the book

This project is Chapter 17 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
