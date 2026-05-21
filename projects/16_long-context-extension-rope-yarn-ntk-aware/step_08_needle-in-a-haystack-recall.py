"""
Project 16: Step 8 — Needle-in-a-haystack recall

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def make_haystack(filler, needle, position_frac, total_len):
    # Build a context of length total_len with the needle
    # inserted at position_frac through the document.
    needle_pos = int(total_len * position_frac)
    tokens = tokenize(filler)[:total_len]
    needle_tokens = tokenize(needle)
    tokens[needle_pos:needle_pos+len(needle_tokens)] = needle_tokens
    return tokens

def needle_test(model, position_fracs, total_lens):
    results = {}
    for L in total_lens:
        for frac in position_fracs:
            ctx = make_haystack(filler, "The secret number is 1729.",
                                frac, L)
            prompt = ctx + tokenize(" What is the secret number?")
            answer = model.generate(prompt, max_new_tokens=10)
            results[(L, frac)] = "1729" in answer
    return results
