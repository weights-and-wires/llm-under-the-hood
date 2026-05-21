# Project 4: Attention From Scratch

## Hook

Why does a model suddenly stop acting like a bag of isolated token IDs and start acting like it can connect pronouns to names, match brackets, continue code blocks, and stay on topic across a paragraph? If each token is just an integer turned into a list of numbers, where does the actual context-sharing happen? That question lands you on attention, because attention is the first place in the model where one token gets to look at the others and decide, in real time, which ones matter.

The first time I built one of these by hand, I spent an hour staring at a shape mismatch error before I realized I had transposed the wrong axis on K. Attention is not conceptually hard. It is operationally fiddly. The mental model is one paragraph long. The code that implements that paragraph involves enough reshape, transpose, and broadcast that a careless line lands you somewhere you cannot debug from the error alone.

## The Concept

Up to now, you have tokens and embeddings. Useful, but still not enough for context. If the token `"bank"` appears, the model has a problem. Does it mean a financial institution? A river bank? A verb, like banking a plane? The token by itself cannot settle that. Attention is the mechanism that lets a token ask the rest of the sequence for help.

Imagine every token in a sentence sitting around a table with an index card. On the card, each token writes three things:

- a **query**: what I am looking for
- a **key**: what I contain
- a **value**: what I can contribute if someone picks me

If you are the token `"it"` in the sentence "The server returned an error because it timed out," you need to figure out what `"it"` refers to. So `"it"` sends out its query: "I am looking for something singular, nearby, and probably a thing that can time out." Other tokens present their keys. `"server"` has a key that says, roughly, "infrastructure noun, singular, plausible agent of timing out." `"error"` has a key that says, roughly, "noun, singular, but less likely to time out." Attention compares the query from `"it"` against the keys from all tokens. The better the match, the more weight that token gets. Then `"it"` forms its new representation by taking a weighted mixture of the values from the tokens it attended to.

That weighted mixture is the whole trick. It does not copy one token exactly. It blends information from the places it thinks matter. So if `"it"` gives `"server"` a large weight and `"error"` a smaller one, the new representation of `"it"` now carries context that helps the model resolve the reference.

The core move is simple: each token asks what it needs, each token advertises what it has, each token decides who matters, and each token updates itself using that information.

This is why attention feels different from the earlier models you built. Earlier, information moved through fixed pathways. Attention creates dynamic ones. The path depends on the sentence in front of you. `"bank"` in one sentence will attend differently than `"bank"` in another. Nothing in the code says, "if river, then use river meaning." The model learns patterns that make that behavior possible.

The first confusion people have is "Is attention memory?" Not really. It is not long-term storage. It is a way of routing information inside the current sequence: for this token, right now, which other tokens should influence the computation? A better picture is a room full of people briefly consulting each other before answering. There is no notebook being written to, no long-term archive. Just one round of "let me check with the others" before each token updates its representation.

Make the analogy concrete. Each token starts as a vector of size `d`. From that vector, the model computes three new vectors:

- `Q`: the query
- `K`: the key
- `V`: the value

All three come from learned linear projections. Three small matrix multiplies that turn the token representation into three different "views" of itself. Why three? Because "what I am looking for," "what I contain," and "what I contribute" are not the same job. A token might be very good at matching a query but not be the main thing you want to copy into the result. A pronoun might look for nouns with one pattern, while the actual information it needs to gather lives in a different subspace of numbers. If a single score had to do all those jobs at once, the model would lose a great deal of expressive flexibility.

The cleanest mental model I have for the Q/K/V split is from a library: each book on the shelf has a title (the key), what you write on the back of the request slip (the query), and what you actually take home (the value). Those are obviously three different things. The librarian does not match request slips to take-home contents directly; the slip matches the title, then the title delivers the contents. Attention is doing the same routing trick at every layer.

Now the math.

For one head of attention, the formula is:

```text
Attention(Q, K, V) = softmax((QK^T) / sqrt(d)) V
```

Here is what every symbol is.

- `Q` is the matrix of queries, one query vector per token
- `K` is the matrix of keys, one key vector per token
- `V` is the matrix of values, one value vector per token
- `K^T` means transpose, which turns rows into columns so dot products line up
- `QK^T` gives a score between every token's query and every token's key
- `d` is the size of each head's query and key vectors
- `sqrt(d)` is the scaling factor
- `softmax(...)` turns raw scores into weights that sum to 1
- multiplying by `V` forms the weighted sum of value vectors

It looks dense until you say it out loud: take every token's query, compare it with every token's key, turn those comparisons into weights, use those weights to mix together the value vectors. That is attention.

The attention matrix itself is worth staring at. If your sequence length is `T`, then `QK^T` is a `T x T` matrix. Row `i` tells you what token `i` attends to. Column `j` tells you how much other tokens attend to token `j`. If row 5 places most of its weight on columns 3 and 4, token 5 is mostly consulting tokens 3 and 4. That is why attention heatmaps are so useful. They let you watch context routing happen.

From some low-level work on Apple Neural Engine kernels, I used to keep printed attention heatmaps taped to the side of my monitor. Not because they helped optimize the kernels, but because they kept reminding me what the kernels were for. It is surprisingly easy to drift into "this is just a big matrix multiply followed by a softmax" and lose the thread that this is the thing that lets the model carry information across positions.

Now the next problem. If you let each token attend to every other token, then during training a token could look at the future. That would be cheating. In an autoregressive model, the model predicts the next token one token at a time, so token 5 can use tokens 1 through 5, but not token 6. If token 5 could attend to token 6 during training, the model could peek at the answer. That is why we need **causal masking**.

Causal masking blocks attention to future positions. The upper triangle of the attention score matrix gets replaced with negative infinity before softmax. Negative infinity sounds dramatic, but the reason is simple: softmax turns very negative numbers into weights extremely close to zero, so future positions become impossible to attend to.

One more step. Single-head attention works, but one head can only learn one pattern at a time. Real language needs several patterns at once. One head may track the previous token. Another head focuses on opening brackets. A third learns subject-verb agreement, while a fourth catches line indentation in code. So transformers use **multi-head attention**. Instead of doing one attention computation over the full embedding dimension, the model splits the embedding into `H` smaller slices. Each slice gets its own query, key, and value projections. Each head computes attention independently. Then the results are concatenated back together.

Think of it as a panel of specialists. Instead of one person trying to watch syntax, meaning, punctuation, and code structure all at once, each specialist gets a smaller view, does one job, and the system combines the reports. That matters because language is not one relation. It is many relations stacked on top of each other. Multi-head attention gives the model multiple ways to look at the same sequence at the same time. That is the first real answer to "How is it keeping track of all this?" Not with one giant understanding module, but with many learned attention patterns running in parallel over token representations that update layer after layer. Figure 4.1 illustrates why the causal mask is essential to this process, and Figure 4.2 shows the full attention pipeline from query scoring through value mixing.

![Figure 4.1. Attention is useful only when the causal mask blocks future positions during autoregressive training; otherwise later tokens leak into earlier predictions.](figures/fig_attention_causal_mask.png)

![Figure 4.2. Attention is a pipeline: the query scores keys, masking blocks illegal positions, softmax turns scores into weights, and those weights mix the value vectors into the next hidden state.](figures/fig_attention_score_softmax_mix.png)

## Why It Matters

Without attention, tokens do not get to dynamically ask "who should matter to me right now?" That removes the main mechanism that makes transformers good at context. Think about prose: "Maria handed the package to Jordan because she was leaving." Who is "she"? You do not solve that by looking at `"she"` alone. You solve it by comparing it against earlier nouns and the surrounding verb structure. A model without attention cannot make that comparison at all. It can only use fixed, position-independent features.

Attention also matters because it is where several critical design choices live: the `1/sqrt(d)` scaling factor, causal masking, multi-head splitting, and softmax over scores. If you break any of these, the model stops behaving well in specific, diagnosable ways. That is exactly why this chapter asks you to inspect the attention matrix, see the weights, and break it on purpose. Not because the formula is complicated, but because each piece of the formula exists to prevent a concrete failure, and you will understand those pieces much better by watching the failures happen than by reading about them.

Strong opinion: most published explanations of attention overweight the math and underweight the breakages. The math fits on a napkin. The breakages are where the engineering lives.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/04_attention-from-scratch/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/04_attention-from-scratch/build.py --full

# The BREAK IT experiment:
python projects/04_attention-from-scratch/break_it.py
```

## Outputs

_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._

## Read in the book

This project is Chapter 4 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
