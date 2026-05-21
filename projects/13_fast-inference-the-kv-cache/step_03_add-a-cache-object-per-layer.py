"""
Project 13: Step 3 — Add a cache object per layer

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class KVCache:
    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

k_cache[layer]  # [B, H, T_max, d_head]
v_cache[layer]  # [B, H, T_max, d_head]

k_cache[layer][:, :, pos:pos+1, :] = k_new
v_cache[layer][:, :, pos:pos+1, :] = v_new
