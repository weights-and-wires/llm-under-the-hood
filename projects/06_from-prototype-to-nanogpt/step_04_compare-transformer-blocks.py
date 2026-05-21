"""
Project 6: Step 4 — Compare transformer blocks

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

x = x + self.attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))
