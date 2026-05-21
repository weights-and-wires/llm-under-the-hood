"""
Project 15: Step 2 — Separate query head count from KV head count

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

self.n_head = 8

self.n_head = 8      # query heads
self.n_kv_head = 2   # shared key/value heads

self.group_size = self.n_head // self.n_kv_head
