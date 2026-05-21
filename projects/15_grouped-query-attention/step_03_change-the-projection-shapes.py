"""
Project 15: Step 3 — Change the projection shapes

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

self.q_proj = nn.Linear(d_model, d_model, bias=False)
self.k_proj = nn.Linear(d_model, d_model, bias=False)
self.v_proj = nn.Linear(d_model, d_model, bias=False)

self.q_proj = nn.Linear(d_model, self.n_head * self.d_head, bias=False)
self.k_proj = nn.Linear(d_model, self.n_kv_head * self.d_head, bias=False)
self.v_proj = nn.Linear(d_model, self.n_kv_head * self.d_head, bias=False)
self.o_proj = nn.Linear(self.n_head * self.d_head, d_model, bias=False)
