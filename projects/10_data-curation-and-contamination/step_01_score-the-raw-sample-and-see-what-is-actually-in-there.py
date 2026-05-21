"""
Project 10: Step 1 — Score the raw sample and see what is actually in there

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

from datasets import load_dataset
import re
from collections import Counter

ds = load_dataset("HuggingFaceFW/fineweb-edu",
                  name="sample-10BT", split="train", streaming=True)
sample = list(ds.take(100_000))

def doc_stats(text: str) -> dict:
    chars = len(text)
    tokens = text.split()
    n_tokens = len(tokens)
    mean_word_len = (sum(len(t) for t in tokens) / n_tokens
                     if n_tokens else 0.0)
    fives = [tuple(tokens[i:i+5]) for i in range(n_tokens - 4)]
    counts = Counter(fives)
    repeated = sum(c for c in counts.values() if c > 1)
    repetition = repeated / max(len(fives), 1)
    return {
        "chars": chars,
        "tokens": n_tokens,
        "mean_word_len": mean_word_len,
        "repetition": repetition,
    }

stats = [doc_stats(d["text"]) for d in sample]
