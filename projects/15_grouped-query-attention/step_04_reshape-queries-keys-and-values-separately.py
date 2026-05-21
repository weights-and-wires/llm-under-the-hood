"""
Project 15: Step 4 — Reshape queries, keys, and values separately

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

k = self.k_proj(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)
v = self.v_proj(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)
