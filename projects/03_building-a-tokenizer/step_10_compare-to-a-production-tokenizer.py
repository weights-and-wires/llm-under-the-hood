"""
Project 3: Step 10 — Compare to a production tokenizer

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import tiktoken

enc = tiktoken.get_encoding("gpt2")
print(enc.encode("replaying the song"))
