"""
Project 18: Step 1 — Isolate the FFN boundary

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

x = x + self.attn(self.norm1(x))
x = x + self.ffn(self.norm2(x))
