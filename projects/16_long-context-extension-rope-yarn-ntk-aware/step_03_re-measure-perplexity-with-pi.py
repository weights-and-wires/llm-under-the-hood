"""
Project 16: Step 3 — Re-measure perplexity with PI

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

model.set_rope(RopePI(head_dim=64, base=10000.0,
                     train_seq_len=1024, target_seq_len=4096))
pi_ppl = {}
for L in context_lengths:
    pi_ppl[L] = evaluate_perplexity(model, eval_dataset, seq_len=L)
    print(f"PI: length={L}, ppl={pi_ppl[L]:.2f}")
