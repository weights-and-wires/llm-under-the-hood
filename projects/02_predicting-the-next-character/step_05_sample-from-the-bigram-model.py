"""
Project 2: Step 5 — Sample from the bigram model

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

g = torch.Generator().manual_seed(42)

ix = stoi["."]
out = []

while True:
    probs = P[ix]
    ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
    ch = itos[ix]
    if ch == ".":
        break
    out.append(ch)

print("".join(out))
