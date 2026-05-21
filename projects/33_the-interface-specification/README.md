# Project 33: The Interface Specification

## Hook

Why does a specialist model that scores well on its own turn into nonsense the moment you plug it into a shared system? Not because it forgot language. Not because the router is weak. Not because the checkpoint is corrupted.

The ugly answer is simpler: both sides think they are speaking the same internal language, but one side quietly switched the wiring. One model hands over a 768-number hidden state that has been normalized one way, rotated in position one way, split into heads one way, and interpreted over one vocabulary. The other model expects a different contract.

From the outside, the tensors fit. From the inside, the meanings do not. This is the kind of failure that survived three of my own long fine-tuning sweeps before I admitted the bug was not in the model.

## The Concept

Think about USB. A USB-C cable looks simple because the mess is hidden in the specification: voltage ranges, pin layout, protocol negotiation, data lanes, power roles. If that specification did not exist, every device maker would ship a connector that looked almost right and failed in slightly different ways. Your phone might charge but not transfer data. Your monitor might flicker only at one resolution. Your keyboard might work until it goes through a hub.

Those are the worst failures. Not total failure. Partial failure that wastes hours and survives every shape check you can think of.

Composable model systems need the same thing. In **Project 32: Fusing Independently Trained Specialists**, we saw that specialists compose only when they share an internal coordinate system. That phrase can sound vague, so let's pin it down.

A coordinate system here means the rules that give meaning to a hidden state. A hidden state is the list of numbers passed from one layer of a model to the next. If `d_model = 768`, then each token is represented at that boundary by 768 floating-point numbers. Those numbers are not random. Their scale matters, their ordering matters, the normalization applied to them matters, the positional encoding scheme matters, the way attention heads are arranged matters, and the vocabulary matters too if you expect outputs to line up.

If two specialists share weights up to some boundary, then branch off, then later get recombined, you need a formal promise about that boundary: what shape comes in, what shape goes out, what normalization has already happened, what positional encoding was assumed when those numbers were produced, what attention head layout the downstream module expects, and what vocabulary the logits refer to. Without that promise, "compatible" becomes guesswork. The first time I watched a cooperative fusion run produce fluent-but-wrong outputs, I assumed for a full afternoon that the issue was the router. It was not. It was a one-character difference in how two specialists declared their normalization.

Here is the analogy that makes this click. Imagine three translators working for the same newsroom. They all receive a note from the editor that says: `BANK CLOSED AFTER FLOOD`. One translator assumes "bank" means financial institution. Another assumes riverbank. A third assumes the note already includes location metadata from earlier in the conversation. All three can produce fluent English. All three can work well alone. But if translator A writes the first half of a paragraph and translator B continues without sharing assumptions, the combined output can be nonsense.

The problem is not grammar. The problem is incompatible hidden context. That is what an interface specification fixes: at this boundary, "bank" has to arrive in this format, with this context encoding, under these assumptions.

Now the engineering version. A versioned interface specification for composable specialists is a written contract that says:

- Hidden state dimension: `d_model = 768`
- Normalization type: `RMSNorm`
- Norm epsilon: `1e-5`
- Residual convention: pre-norm or post-norm
- Attention head configuration: `H = 12`, head dimension `d_head = 64`
- Positional encoding: `RoPE` with specific base and scaling
- Vocabulary: exact tokenizer and vocabulary file hash
- Output boundary: logits over vocabulary `V = 50,304` or hidden state only
- Checkpoint metadata: version string, architecture fingerprint, training origin

This sounds bureaucratic until you see the alternative. The alternative is silent garbage.

A model with `LayerNorm` at the specialist boundary can often still produce tensors of the correct shape. No crash, no red warning, no obvious stack trace. The numbers even look normal if you inspect mean and standard deviation. But they no longer mean the same thing as the tensors expected by a system built around `RMSNorm`. That is the core lesson of this chapter: shared weights are not enough. Shared contracts are what make composition dependable.

I will be honest about a bias here. Years of federal-procurement work at Procore, where FedRAMP and SOC 2 governance treat every interface as a written contract subject to a Change Advisory Board, have made me allergic to "we just trust the shape." It is not paranoia. It is the only style of work I have seen survive auditors and time. Figure 19.1 shows the contract structure that enforces this explicitly.

![Figure 19.1. An interface specification makes compatibility explicit by checking normalization, positional encoding, head layout, and tokenizer identity instead of trusting tensor shape alone.](figures/fig_interface_spec_contract.png)

## Why It Matters

Without an interface specification, you cannot tell the difference between these four situations: the specialist is genuinely bad; the specialist is good but the fusion system is bad; the specialist and fusion system are both good but the boundary assumptions differ; or the system mostly works only because one accidental mismatch has not bitten you yet.

That ambiguity destroys engineering progress. If a specialist plugs into the fusion system and outputs garbage, what do you do next? Retrain longer? Change the router? Add more data? Tune the learning rate? Replace the expert? Without a contract and a compliance test, you are debugging in the dark. I have wasted whole afternoons on the wrong layer of this question, and the lesson was always the same.

This matters even more once multiple people work independently. Suppose one person trains a code specialist, another trains a multilingual specialist, a third builds the fusion stack, and a fourth writes evaluation. If each person makes one "reasonable" architectural choice on their own, you can end up with four modules that all work in isolation and fail together.

Worse, they may fail softly. Perplexity rises a bit. Accuracy drops only on long contexts. Responses become vague instead of obviously broken. Those are the failures that survive demos and poison production. In a past role on a multi-state SaaS platform, the multi-party API integrations across DMV, DCF, banking, and payment systems were all "compatible" on shape and protocol. The failures that cost real money were always semantic. A field that meant dollars in one system meant cents in another, a date that meant local time on one end meant UTC on the other, and nothing in the wire format would tell you.

A good interface specification gives you three things: coordination without constant meetings, fast rejection of incompatible checkpoints, and blame assignment when something breaks. That last one matters. If the compliance test says:

- expected `RMSNorm`, found `LayerNorm`
- expected `rope_theta=10000`, found `rope_theta=500000`
- expected tokenizer hash `abc123`, found `e91f2d`

then the mystery evaporates. You are not "tuning composition." You are fixing a contract violation. Same reason APIs have versions, schemas, and compatibility checks. An API that returns a JSON object where `price` silently changes from dollars to cents without changing the version number is broken even if the endpoint still returns `200 OK`. I lived this exact bug on an enterprise healthcare eligibility integration; the endpoint was happy, the auditor was not. A specialist that silently changes boundary normalization is broken in the same way. The spec is not paperwork. It turns invisible incompatibility into an engineering error you can see.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/33_the-interface-specification/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/33_the-interface-specification/build.py --full

# The BREAK IT experiment:
python projects/33_the-interface-specification/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 33 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
