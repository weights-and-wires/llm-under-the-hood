"""
Project 2: Step 3 — Bigram counts

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import torch

N = torch.zeros((V, V), dtype=torch.int32)

for word in words:
    chs = ["."] + list(word) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        i = stoi[ch1]
        j = stoi[ch2]
        N[i, j] += 1
