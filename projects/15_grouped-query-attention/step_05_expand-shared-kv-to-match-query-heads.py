"""
Project 15: Step 5 — Expand shared K/V to match query heads

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, H_kv, T, d_head = x.shape
    x = x[:, :, None, :, :].expand(B, H_kv, n_rep, T, d_head)
    return x.reshape(B, H_kv * n_rep, T, d_head)

k = repeat_kv(k, self.group_size)
v = repeat_kv(v, self.group_size)
