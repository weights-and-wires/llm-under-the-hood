"""
Project 3: Step 9 — Build a token visualizer

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

tokens = tokenizer.encode("replaying the song")
for tid in tokens:
    piece = tokenizer.vocab[tid].decode("utf-8", errors="replace")
    print(f"[{piece}]")
