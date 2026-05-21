"""
Project 3: Step 5 — Build encode and decode

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def decode(ids: list, vocab: dict) -> str:
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")

def encode(text: str, merges: dict) -> list:
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = {(a, b): i for i, (a, b) in enumerate(zip(ids, ids[1:]))}
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids
