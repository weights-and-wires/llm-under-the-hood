"""
Project 3: Step 2 — Count adjacent pairs

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

[104, 101, 108, 108, 111]

from collections import Counter

def get_stats(ids: list) -> Counter:
    counts = Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts
