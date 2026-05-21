# Project 12: Distributed Training: FSDP and ZeRO (Single-Box Proxy)

## Hook

Every project so far fit on one GPU. The model weights lived in one place, the gradients lived next to them, the optimizer state lived next to that, and the only thing moving across a network was your `git push`. Now imagine a model whose weights alone would not fit in any single GPU on the planet. A 70-billion-parameter network, say, whose parameters, gradients, and Adam optimizer state together need about 1.1 terabytes of memory. How does that train at all? Where does each piece live, and how do the pieces talk to each other?

Before we go any further: this chapter teaches the mechanics on a **single-box proxy**. Two PyTorch processes on one GPU, or two CPU processes on a laptop. Most readers do not have an 8xH100 cluster handy, and the patterns we are about to build transfer exactly to multi-node training without you ever needing one. The proxy lab is the real lab. The bigger version is the same code with a different launcher.

I want to flag this up front because when I drafted this chapter, I tried writing it the other way around. Cluster pseudocode first, then a smaller proxy. Reading it back, the chapter sounded like it was lecturing about hardware most readers will never own. I scrapped that version. The proxy comes first now because it is the version you will actually run.

## The Concept

Think of a library where every member owns three or four books, not the whole collection. When someone needs to read a book, they ask the member who owns it and they pass it over. The reader reads, hands the book back, and the owner puts it on their shelf again. Nobody hoards the catalog. Storage is cheap because each member only carries their share, and the only cost is that books move around when they are needed.

That picture is the entire idea behind **Fully Sharded Data Parallel** training, usually shortened to **FSDP**. The model's parameters get cut into N pieces, one per worker process, and so do the gradients and the optimizer state. Each worker owns exactly 1/N of each, and only 1/N. To run a forward pass, the workers gather the pieces they need from each other just in time, run the layer, and then drop the gathered copies. The full weight tensor exists for a few milliseconds during forward and a few more during backward, and then it disappears again. The rest of the time, each worker carries only its slice.

A few terms before we go further, because we are about to use them constantly.

A **worker** or **rank** is one process running one copy of the training script. In multi-GPU training, one rank usually owns one GPU. In our single-box proxy, two ranks share one GPU (or two ranks each own a CPU). The set of ranks participating in a training run is called the **world**, and the total count is the **world size**. Rank 0 through rank N-1 each have a unique integer identity.

A **shard** is a slice of a parameter tensor. If a weight matrix has shape `(1024, 4096)` and the world size is 4, the simplest way to shard is to flatten the tensor to length `1024 * 4096 = 4_194_304` and give each rank a contiguous slice of length `1_048_576`. Rank 0 owns the first slice, rank 1 the second, and so on. Sharding works on the flat view because flat slicing always divides cleanly with padding, while slicing a multi-dimensional tensor along, say, the row axis runs into edge cases when the row count is not divisible by the world size.

The first time I worked on a sharded system at this scale was nothing like an LLM. It was the Aspira AWS migration. Customer data spread across 28 parks and roughly 5,000 sites, six years of campground reservations, ported into Redshift without losing a record at 99.99% uptime. The math of FSDP is different, the stakes are different, the bytes are different. The mental discipline is the same one. Stop thinking about the data as "one big object that you address by its full shape." Start thinking about it as "N consistent slices that have to agree about who owns what." Half of distributed engineering is just internalizing that shift.

**All-gather** is the operation that collects the full tensor from all ranks. Every rank starts with its 1/N slice. Every rank ends holding the full concatenated tensor. The cost is communication: each rank sends its slice to every other rank, or equivalently the system routes the slices through whatever interconnect is fastest. The dual operation is **reduce-scatter**, which goes the other direction. Every rank starts with a full tensor (typically a full gradient), the system sums those tensors element-wise across all ranks, and then each rank ends up with a 1/N slice of the summed result. All-gather is "spread the shards out into the whole." Reduce-scatter is "collapse the whole into shards while summing."

The reason to know the names is that these two operations are the entire vocabulary of FSDP. Forward pass uses all-gather to materialize each layer's weights just before computing the layer. Backward pass uses reduce-scatter to fold per-rank gradients back into sharded gradients. That is it. Everything else is bookkeeping.

**Zero Redundancy Optimizer**, almost always called **ZeRO**, is the same idea told as a three-step ladder. ZeRO Stage 1 shards only the optimizer state, that big chunk of fp32 Adam moments that lives next to the parameters and is twice as large as the parameters themselves. ZeRO Stage 2 adds gradients to what gets sharded. ZeRO Stage 3 adds the parameters themselves. ZeRO-3 and FSDP are the same algorithm wearing different brand names. DeepSpeed shipped ZeRO first, PyTorch shipped FSDP later, and the two communities use both names for the same picture.

The savings curve is the part that makes this worth the engineering. If you have N workers and you fully shard parameters, gradients, and optimizer state, each worker holds 1/N of the storage. With AdamW, a model of P parameters in mixed precision needs roughly `2P + 2P + 12P = 16P` bytes: two bytes per parameter for the bf16 weights, two bytes per parameter for the gradients, four bytes for the fp32 master weights, and eight bytes for the two fp32 Adam moments. For a 70-billion-parameter model that is about 1.1 terabytes. With 64 workers fully sharded, each worker holds about 17 gigabytes, which fits in one H100. The model that did not fit on any single GPU fits on every single GPU at the same time, with room to spare.

The first time I sat with these numbers, I expected the storage savings to be the interesting part. They are not. The interesting part is the second-order consequence: once each worker only holds 1/N, the cost of starting and ending each layer goes up, because somebody has to materialize the full tensor for the few microseconds the kernel needs it. The savings are real. They are also paid for. The rest of this chapter is mostly about reading both sides of that ledger at once.

![Figure 12.1. A single parameter tensor sharded across N ranks. The flat view is sliced into N contiguous pieces of equal length; each rank owns one piece and nothing else.](figures/fig_fsdp_shard_layout.png)

The catch is communication. The model still has to assemble itself layer by layer to compute. If your interconnect is fast (NVLink between GPUs in one node, NVSwitch between nodes), the cost is small enough that the trade pays off. If your interconnect is slow (generic ethernet between cloud machines, with millisecond round-trip times), the cost can dominate, and the model spends more time waiting on the network than computing. Real FSDP deployments live on this knife edge, and a serious part of the job is reasoning about where the time goes.

A small caveat on terminology, because the literature will confuse you otherwise. People sometimes say "data parallel" to mean what we used to do, where you replicate the model on every worker and split the batch. They say "fully sharded data parallel" to mean what this chapter teaches. The word "data" in both names refers to splitting the input batch across workers, which is still happening here. The "fully sharded" part is the new piece: even though every worker processes a different batch slice, the model parameters themselves are no longer replicated. They are sharded.

The mental shift the rest of the chapter pushes is this. In ordinary single-GPU training, every parameter lives in one place at all times, and computation is local. In FSDP, every parameter lives in one place most of the time and in every place during the few microseconds when its layer is computing. Memory is the resource you are saving. Communication is the resource you are paying. The skill is reading both at once.

Most introductions to distributed training I have read are honestly bad on this point. They walk you through the API. They leave you fluent in syntax and blind to the budget you are spending. The whole job is the budget.

## Why It Matters

Without FSDP or ZeRO, training models larger than a single GPU's memory becomes either impossible or absurdly expensive. You can rent a GPU with 80 gigabytes of memory, but you cannot rent one with 1.1 terabytes, because no such product exists. Above a certain size, sharding stops being a clever optimization and starts being the only way the run happens at all. Every frontier model trained in the last three years used some form of this idea.

The second reason this chapter matters is debugging. Distributed bugs are different from single-GPU bugs. On one GPU, when something is wrong, it crashes or produces garbage immediately. On multiple ranks, it can crash on one rank and not the others, hang forever because two ranks are waiting on a collective the third rank never reached, or, worst of all, silently diverge, where each rank trains a slightly different model and the loss curves look fine on each rank but the eventual checkpoint is nonsense. We will see one of those silent failures up close in the BREAK IT section.

There is a third reason, less direct but worth naming. Reading distributed training code is a skill the field has decided is optional, which means most engineers cannot do it. If you can look at a 200-line FSDP wrapping function and tell which calls are all-gathers and which are reduce-scatters, you have a leg up on most of the people working in this area. The mechanics are not hard. The notation is unfamiliar. This chapter fixes the notation problem.

I will say it plainly because I felt this for years. The barrier to entry on distributed training is almost entirely vocabulary, and the vocabulary is poorly taught.

The skill you build here also feeds directly into later chapters. **Project 17: Production Serving — vLLM, Continuous Batching, Quantization** uses tensor parallelism, a sibling technique that shards along a different axis. **Project 18: Mixture of Experts** runs expert-parallel layers, which is yet another sharding pattern. **Project 23: Reward Models and RLHF** trains a policy and a reward model concurrently, sometimes on the same hardware, and the memory-balancing math comes straight out of the ZeRO ladder. The vocabulary you learn here (rank, world size, shard, all-gather, reduce-scatter) recurs in every one of those chapters.

A final reason: the things that go wrong in distributed training are exactly the things instrumentation from **Project 11: Training Debugging — Spikes, NaNs, Profiling** lets you catch. The four vital signs from that chapter (loss, gradient norm, activation distribution, and update ratio) all behave a little differently across ranks during a failure, and a per-rank dashboard is the single best diagnostic when distributed training goes wrong. We will sketch what that looks like in Step 8.

Some of this is grounded in our own work. The KALAVAI cooperative-training paper (arXiv:2603.22755) covers fusion across 4 model families from 410M to 6.9B, more than 7,550 individual runs, with +7.7% perplexity improvement averaged across families. Every one of those runs lived or died on per-rank sanity checks. Without that hygiene, the loss curves look believable on each contributor's machine and the fused model is garbage. The reason this chapter exists in its current form is that the lessons from those runs are not in the textbooks.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/12_distributed-training-fsdp-and-zero-single-box-proxy/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/12_distributed-training-fsdp-and-zero-single-box-proxy/build.py --full

# The BREAK IT experiment:
python projects/12_distributed-training-fsdp-and-zero-single-box-proxy/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 12 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
