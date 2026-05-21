# Project 34: Incremental Assembly

## Hook

Why can't you just bolt a new specialist onto a composed model the way you install a new plugin into an app? You already have the specialists, the router, and the hidden-state contract.

So why does adding one more expert after deployment so often make the system worse at old tasks, mediocre at the new task, or dependent on the order you attached things in?

I will be blunt. This is the question I think the field has been quietly underselling. Everything else in modular LLM work is rehearsal for it.

## The Concept

Start with a company, not a transformer. You have three consultants on retainer: legal, tax, and technical architecture. Every incoming question goes through a receptionist who decides where it should go. It works until you hire a fourth consultant for privacy compliance. You do not want to retrain the old consultants, and you do not want to rebuild the whole company every time someone new arrives. So you tell the receptionist, "questions about privacy go to the new person."

That sounds like a local change. It is not.

The receptionist learned boundaries among the specialists that already existed, not an abstract concept of privacy compliance. Routing was shaped by the specific contrast between legal, tax, and architecture, not by some general theory of what a specialist is. Adding a fourth choice does not just expand the menu; it reshapes the relative boundaries between every option on it.

That is the central problem of incremental assembly. A composable system is not just several specialists wired together. New specialists have to join later without forcing the coordination mechanism to relearn the world. Figure 20.1 captures this challenge structurally.

![Figure 20.1. Incremental assembly asks a stricter question than simple fusion: can specialist D join an A/B/C system later without degrading the old domains or making results depend on installation order?](figures/fig_incremental_assembly_pipeline.png)

Project 32 gave us the first version of fusion: multiple specialists combined behind a routing layer. Project 33 made us write down the boundary contract so we stop pretending "same tensor shape" means "same internal language." Project 34 now asks the ugly question: can you add new specialists after deployment and keep the system stable?

This is the question my own work on cooperative LoRA fusion was designed to push on. The protocol I drafted spent more design effort on "what does post-hoc specialist addition need to look like" than on the fusion math itself. The fusion math is the easy part. The protocol around it is where the actual research lives.

### What "incremental assembly" means

Incremental assembly means you already have a composed system with several specialists, you train a new specialist independently, you attach that specialist later without retraining all specialists, ideally without retraining the router either, and you expect old behavior to remain stable while new behavior appears where expected. And results should not depend much on the order you added specialists. If specialists A, B, and C behave differently under different installation orders, the system is path-dependent.

### Why This Is Harder Than It Sounds

The usual mistake is to treat the router like a lookup table. It is not. It is a learned decision mechanism. Given a hidden state (a list of numbers representing the token's current meaning and context), it chooses where that token should go next. So the router has to recognize what kind of input it is seeing, know which specialist can help, and keep doing both of those things after you expand the set of choices. That third requirement is where things break.

Distinguishing among 3 choices is not the same problem as distinguishing among 12, and distinguishing among 12 specialists where two are nearly identical is harder still.

Sorting email into three folders is easy. Work, personal, spam. It is harder to sort into twelve narrowly defined folders, and gets much harder when two folders differ only by a subtle line like "database scaling" versus "distributed data infrastructure." You will misroute more often. Routers hit the same wall. I keep my own inbox in three folders for exactly this reason; I have tried twelve and the routing always collapsed back to three within a month.

### The Hidden Assumption Everyone Makes

Most model-composition work assumes the specialists are present during training, so the router learns relative boundaries. Not "what privacy compliance is in an absolute sense," but "this region looks more like specialist 4 than specialist 1, 2, or 3."

Adding a new specialist later is not a local change. You changed the choice set. You added a new answer to the exam. Even if the old answers were correct before, the decision boundary can shift, because "just add one more specialist" changes the competition.

That is the scaling limit this project is after. Not whether specialists help in principle, but whether the routing mechanism can still tell them apart as the menu grows.

### The Minimum Math You Actually Need

Let the router produce a score for each specialist. Call those scores `s_1, s_2, ..., s_M`, where `M` is the number of specialists and each `s_i` is "how suitable does the router think specialist `i` is for this token?" The router often turns those scores into probabilities with softmax:

```text
p_i = e^(s_i) / sum_(j=1 to M) e^(s_j)
```

In plain English: `p_i` is the router's confidence for specialist `i`, `e^{s_i}` means "make bigger scores count much more," and we divide by the total so all probabilities add up to 1. The reason to care about this formula is simple: add more specialists and the denominator gets larger. If the scores are not sharply separated, the probabilities flatten. Flattened probabilities mean weaker discrimination. That is one reason routing gets worse as the menu grows.

## Why It Matters

If incremental assembly works, you no longer rebuild the whole model every time you want one new capability. You can imagine adding a legal specialist next quarter, a multilingual specialist after that, a private internal specialist for one customer, all while preserving the old system's behavior. This is software engineering logic applied to models: stable interfaces, modular upgrades, bounded changes.

A multi-state SaaS platform I worked on for eight years grew by exactly this pattern at the integration layer. We added DMV connectors, then DCF connectors, then new banking and payment integrations, then a fresh state's compliance rules, without ever rebuilding the core. The reason was not luck. It was that the early team had treated each integration as a contract first and a feature second. Model systems are the same problem with worse tooling.

### Without incremental assembly, "modular" is mostly branding

A system is not modular because its diagram has boxes and arrows. If every new specialist forces router retraining, or worse, full-system retraining, you do not have modularity. You have coordinated co-training, which is useful but different.

A system that needs global retraining for every specialist addition takes longer to update, costs more to maintain, becomes harder to reason about, makes regressions more likely, and cannot support independent specialist release cycles. A system with real incremental assembly lets you treat specialists more like components than entangled organs.

### Why order independence is a serious test

Suppose you add specialists A, B, and C. If `A -> B -> C` gives one result and `C -> A -> B` gives another, then your system is path-dependent. The final behavior depends on historical accident, not just the set of components present.

Order dependence tells you the router or assembly rule is not learning a stable partition of responsibility; it is being nudged by history. That is dangerous because it makes reproduction hard, makes debugging hard, and makes claims about "composable specialists" weaker than they sound.

### Why the scaling limit is the real research question

Early demonstrations often stop at a small number of experts because the picture still looks clean. Three or four specialists is the easy case. The real question is what happens when you keep going.

At some point routing accuracy plateaus, existing specialists interfere, new specialists cannibalize old traffic, confidence spreads too thin, and near-duplicate specialists become almost indistinguishable. That point is not a nuisance result. It is the contribution.

If you can measure where the current approach breaks, you learned something concrete: how much capacity the router has, how separable the domains really are, whether the boundary representation supports modular growth. Research often advances by finding limits with clean measurements, not by pretending the limit is not there. Across the long training sweeps behind one of my own preprints, the most useful single chart in the supplement was the one that mapped where post-hoc addition stopped working cleanly. The headline numbers in the abstract were ornament. The breakdown curve was the actual paper.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/34_incremental-assembly/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/34_incremental-assembly/build.py --full

# The BREAK IT experiment:
python projects/34_incremental-assembly/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 34 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
