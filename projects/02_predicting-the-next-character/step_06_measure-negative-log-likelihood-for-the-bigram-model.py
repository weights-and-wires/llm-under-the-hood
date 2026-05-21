"""
Project 2: Step 6 — Measure negative log-likelihood for the bigram model

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

log_likelihood = 0.0
n = 0

for word in words:
    chs = ["."] + list(word) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        i = stoi[ch1]
        j = stoi[ch2]
        prob = P[i, j]
        log_likelihood += torch.log(prob)
        n += 1

nll = -log_likelihood / n
print(nll.item())
