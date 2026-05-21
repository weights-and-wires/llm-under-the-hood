"""
Project 14: Step 6 — Sweep K from 1 to 8 and find the sweet spot

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

results = {}
for K in [1, 2, 3, 4, 5, 6, 7, 8]:
    tps, accept = measure_speculative_tps(draft, main, prompt, K=K, total=512)
    results[K] = (tps, accept)
    print(f"K={K}: {tps:.1f} tokens/sec, acceptance rate {accept:.3f}")
