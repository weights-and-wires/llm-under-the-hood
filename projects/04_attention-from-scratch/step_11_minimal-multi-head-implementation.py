"""
Project 4: Step 11 — Minimal multi-head implementation

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

# x: (T, d_model)
Q = x @ W_Q
K = x @ W_K
V = x @ W_V

# reshape to (H, T, d_head)
Q = Q.view(T, H, d_head).transpose(0, 1)
K = K.view(T, H, d_head).transpose(0, 1)
V = V.view(T, H, d_head).transpose(0, 1)

scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_head)   # (H, T, T)
scores = scores.masked_fill(mask == 0, float('-inf'))
weights = torch.softmax(scores, dim=-1)
head_out = weights @ V                                   # (H, T, d_head)

# back to (T, d_model)
out = head_out.transpose(0, 1).contiguous().view(T, d_model)
