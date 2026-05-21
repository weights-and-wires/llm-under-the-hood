"""
Project 14: Step 5 — Measure speedup against naive KV-cached generation

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def measure_speculative_tps(draft, main, prompt, K=4, total=256):
    ids = prompt.clone()
    accepted_total = 0
    rounds = 0
    start = time.time()
    while ids.shape[1] - prompt.shape[1] < total:
        new_ids = speculative_step(draft, main, ids, K=K)
        ids = torch.cat([ids, new_ids], dim=1)
        accepted_total += new_ids.shape[1] - 1  # exclude the bonus/residual
        rounds += 1
    elapsed = time.time() - start
    accept_rate = accepted_total / (rounds * K)
    return total / elapsed, accept_rate
