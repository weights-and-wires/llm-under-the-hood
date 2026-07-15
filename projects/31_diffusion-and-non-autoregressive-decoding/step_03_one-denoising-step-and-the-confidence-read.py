"""
Project 31: Step 3 — One denoising step, and the confidence read

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

@torch.no_grad()
def predict(x, fill):
    logits = model(x).logits[0]              # (seq_len, vocab)
    probs = F.softmax(logits[fill], dim=-1)  # (n_blanks, vocab)
    conf, pred = probs.max(dim=-1)           # best prob + best token id
    return conf, pred                        # each shape (n_blanks,)
