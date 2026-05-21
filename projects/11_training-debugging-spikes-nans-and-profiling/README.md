# Project 11: Training Debugging: Spikes, NaNs, and Profiling

## Hook

At step 1,247 of your training run, the loss jumps from 3.2 to 47.8, hangs there for two steps, and then prints `nan`. The training was working, until it wasn't.

The first instinct most people have is to restart with a smaller learning rate and hope it goes away this time. I had that instinct for years. It cost me real money in cluster hours before I learned the other one.

This chapter teaches the other instinct: read the training run like a production system you are on-call for, name the specific failure that just happened, and know which knob to turn before you touch anything. By the end of the chapter you will be able to predict a `nan` several minutes before it arrives, and you will know which of half a dozen specific failure modes caused it.

## The Concept

Picture baking a cake in an oven with no window. You can only check whether the cake is done by opening the door, and opening it too often ruins it. Most people training neural networks debug exactly like this. The training loop is the closed oven. The loss curve is the timer ticking on the wall. If something is wrong inside the oven, you find out at the end, when the cake comes out either burnt or raw.

The fix in both cases is the same. Put instruments inside. Watch the temperature at three points in the oven, not just at the door. Smell the air. Listen for the timer ticking faster or slower than it should. With proper instruments, you do not need to open the door. You can read the state of the cake while it is baking.

A training run has its own set of vital signs. There are four worth measuring continuously, and learning to read all four together is the entire skill this chapter teaches.

I picked up this framing the hard way, before I trained any model. At Aspira I spent years running a multi-state SaaS at a 99.99% uptime target, and the rule there was simple: anything you cannot graph, you cannot debug at 3 AM. Training runs are the same. The difference is that nobody hands you the graphs by default. You have to wire them yourself.

The first vital sign is the **loss** itself. Every training framework prints this number, and almost every reader of this book has stared at one go down. By itself, loss is a lagging indicator. It tells you the model is sick after the model is already sick. By the time the loss curve bends the wrong way, the failure is already a few steps old, and depending on how aggressive it is, it may already be unrecoverable.

The second vital sign is the **gradient norm**, computed layer by layer. For each parameter group in the model (say, each transformer block), take the gradients produced by the most recent backward pass and compute their L2 norm. That is one number per block per training step. Plot it. A healthy run shows the gradient norms drifting slowly, with the lines for different blocks staying roughly parallel. A run heading for a spike has one line (usually the embedding layer's or the first attention block's) pulling away from the cluster and climbing faster than the others. By watching gradient norms instead of waiting for the loss to bend, you catch the spike five to fifteen steps earlier. That is enough time to save a checkpoint or kill a doomed run.

Across thousands of training runs I have logged for cooperative LoRA fusion work on models from a few hundred million to a few billion parameters, this single graph saved more wall-clock than any other instrument. Not because the math is exotic. Because the first attention block almost always leaves the cluster first, and once you have seen that shape three or four times, you stop arguing with the dashboard.

The third vital sign is the **activation histogram** at each layer boundary. After every transformer block, what does the distribution of hidden-state values look like? Healthy activations have a roughly stable spread, a mean near zero if LayerNorm or RMSNorm sits at that boundary, and bounded extremes. Unhealthy activations spike. One feature dimension suddenly carries values in the hundreds, and the softmax that consumes those values collapses into a near one-hot pattern. Once attention saturates, the gradient through that head becomes tiny, and the head stops contributing to learning. The signature of impending attention saturation appears in the activation histogram several steps before the loss notices.

I expected gradient norms to be the most useful of the four when I first wired this dashboard. I was wrong. The activation tail moves earlier, more clearly, and on a smaller dynamic range. Gradient norms are easier to plot and easier to teach. The activation `abs_max` is what I actually look at first when something feels off.

The fourth vital sign is the **weight-update ratio**. At each optimizer step, the optimizer adds some quantity Δw to the current weight w. The ratio |Δw| / |w| for each parameter group tells you how aggressively the optimizer is moving that group relative to its current size. There is a rule of thumb, often credited to Andrej Karpathy: a healthy ratio sits near 1e-3. A ratio of 1e-1 means the optimizer is rewriting that layer every single step. The learning rate is much too hot for that layer. A ratio of 1e-6 means the layer is barely moving. Either the gradient signal is dying, or the layer was already converged and is sitting still. Either way, you want to know.

![Figure 11.1. The four vital signs plotted together for a healthy run. Loss drifts down smoothly. Gradient norms drift in a narrow parallel band. Activation max stays bounded. Update ratios cluster near 1e-3.](figures/fig_training_health_baseline.png)

So far this is bookkeeping. The interesting move is what happens when you start using these four signals together. A loss spike on its own is a fire alarm: useful, but it tells you nothing except that something is burning. The combination of loss, gradient norm, activation distribution, and update ratio is a fire alarm that tells you which floor of the building is on fire, which room, and what is burning. You stop guessing at fixes and start applying targeted ones.

Most training failures fall into a short list of named patterns, and each one has a specific signature across the four instruments.

A **loss spike from a hot learning rate** shows the update ratio drifting upward first, then the activation max, then the gradient norm, then finally the loss, in that order, with the loss arriving last. An **activation explosion from bad initialization** shows the activation max already high at step zero and refusing to come down; the model never enters a healthy state. **Attention saturation** shows one head's contribution to the activation distribution growing while the others shrink; the softmax of that head becomes one-hot, and the head dies. A **mixed-precision overflow** shows a gradient briefly going `inf`, the loss scaler halves itself, and a step gets silently skipped.

A **dead layer** shows one parameter group's update ratio pinned near 1e-6 for thousands of steps while every other group moves normally. This is usually a bad initialization, a wrong learning rate for that group, or, most painfully, a parameter that someone froze and then forgot about. I have shipped that bug twice. Both times the model trained for hours before anyone noticed the frozen block was the one we needed to move the most.

Each one of these patterns has a fingerprint. Once you have seen the fingerprint three or four times, the instruments narrate the failure as it happens. You do not need to wait for the symptom. You name the failure mode while there is still time to clip the gradient, lower the rate, reinitialize the layer, or save the checkpoint before everything blows up.

A small note about what this chapter does not teach. It does not promise to prevent every failure. Some training runs are genuinely impossible: bad data, bad architecture, bad optimizer, hardware that drops bits under load. The job here is narrower: when a run can be saved, you can now save it; when it cannot, you can now name why before the GPU bills mount.

## Why It Matters

Without instrumentation, debugging a training run is mostly theater. You change a hyperparameter, restart, watch the loss for a while, and hope. If the run survives, you keep the change and convince yourself you have learned something. If it does not, you change something else. After a few weeks of this, you have either burned through a research budget without insight or you have given up on training your own models and gone back to fine-tuning APIs that someone else trained.

With instrumentation, the loop becomes diagnostic instead of superstitious. A spike has a cause. The cause has a signature. The signature is visible in the data before the spike lands. The fix is a specific intervention: clip the gradient, lower the rate, reinitialize the bad layer, switch on a mixed-precision scaler. Each intervention has a measurable effect on the next ten or twenty steps, so you can verify whether the fix worked without waiting for the whole run to finish. Training stops being a black box you pray at and becomes a system you can reason about.

There is a second reason this chapter sits where it does. Every subsequent chapter in the book trains a model that can fail. **Project 9: Pretraining on the Real Web** pretrained a model on real web data and could spike if the learning rate was wrong. **Project 12: Distributed Training: FSDP and ZeRO (Single-Box Proxy)** shards parameters across processes, and a mis-shard causes silent divergence, silent except in the gradient-norm trace, which shouts. **Project 23: Reward Models and RLHF** introduces reward hacking, which shows up first as a gradient-norm anomaly in the policy head before it shows up in evaluations. **Project 18: Mixture of Experts** trains a routed model, and router collapse appears in the activation distribution before it appears in the loss. Each of these failure modes has the same shape as the ones in this chapter, only with new layers and new names. The skill transfers exactly.

There is also a third reason worth naming, even though it is less direct. When you can read a training run, you can read other people's runs too. Pull a published tensorboard from a paper repo, look at the gradient-norm curves, and you can often tell whether the run was healthy or just lucky. That is not a small skill. It is the difference between trusting a paper's headline number and understanding the conditions under which the number was produced.

Most published training results do not show the dashboards. I would trust their numbers more if they did. A loss curve without a gradient-norm trace beside it is half a story, and the missing half is almost always the half you would have argued with.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/11_training-debugging-spikes-nans-and-profiling/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/11_training-debugging-spikes-nans-and-profiling/build.py --full

# The BREAK IT experiment:
python projects/11_training-debugging-spikes-nans-and-profiling/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 11 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
