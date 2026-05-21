"""
Project 15: Step 10 — Keep the code mentally sorted into three layers

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

# x: (B, T, d_model)
q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
k = self.k_proj(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)
v = self.v_proj(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)

# Share each KV head across a group of query heads
k = repeat_kv(k, self.n_head // self.n_kv_head)
v = repeat_kv(v, self.n_head // self.n_kv_head)

# Standard scaled dot-product attention after shapes line up
att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
att = att.masked_fill(causal_mask == 0, float('-inf'))
att = F.softmax(att, dim=-1)
y = att @ v
