"""
Project 16: Step 1 — Extend the context window with no fix and watch it break

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

# Naive extension: just feed longer inputs to a model
# whose RoPE was trained at max_seq_len = 1024.
context_lengths = [1024, 1536, 2048, 4096]
baseline_ppl = {}
for L in context_lengths:
    ppl = evaluate_perplexity(model, eval_dataset, seq_len=L)
    baseline_ppl[L] = ppl
    print(f"length={L}, ppl={ppl:.2f}")
