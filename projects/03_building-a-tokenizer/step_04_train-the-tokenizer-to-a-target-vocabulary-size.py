"""
Project 3: Step 4 — Train the tokenizer to a target vocabulary size

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

ids = list(text.encode("utf-8"))
merges = {}
vocab = {i: bytes([i]) for i in range(256)}

for new_id in range(256, V):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    ids = merge(ids, pair, new_id)
    merges[pair] = new_id
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
