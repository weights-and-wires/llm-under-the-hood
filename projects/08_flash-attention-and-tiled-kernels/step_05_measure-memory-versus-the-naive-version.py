"""
Project 8: Step 5 — Measure memory versus the naive version

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def measure_tiled(seq_lens, batch=2, heads=8, dim=64, device="cuda", dtype=torch.float16):
    results = {}
    for n in seq_lens:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        q = torch.randn(batch, heads, n, dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = tiled_attention(q, k, v, BR=64, BC=64)
        torch.cuda.synchronize()
        results[n] = torch.cuda.max_memory_allocated() / 1e9
        del q, k, v, out
    return results

print(measure_tiled([1024, 2048, 4096, 8192, 16384, 32768]))
