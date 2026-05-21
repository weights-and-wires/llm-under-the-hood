"""
Project 7: Step 8 — Replace learned positional embeddings with RoPE

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

x = tok_emb[idx] + pos_emb[pos]

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

def build_rope_cache(seq_len: int, head_dim: int, device, base: int = 10000) -> tuple:
    half_dim = head_dim // 2
    freq_seq = torch.arange(half_dim, device=device).float()
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))

    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, inv_freq)  # (T, head_dim/2)

    cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # (T, head_dim)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos[None, None, :, :], sin[None, None, :, :]

cos, sin = build_rope_cache(T, head_dim, x.device)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
