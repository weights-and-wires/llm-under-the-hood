"""
Project 13: Step 5 — Thread the cache through the full model

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

logits, cache = model.prefill(prompt_ids, cache=None)
next_logits, cache = model.decode(next_token, cache)

logits = model(idx, cache=cache, use_cache=True)
