"""
Project 8: Step 7 — Benchmark wall-clock

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import time

def bench(fn, q, k, v, repeats=10):
    for _ in range(3):  # warmup
        fn(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(q, k, v)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats
