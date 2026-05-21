"""
Project 16: Step 5 — Re-measure perplexity and compare PI vs NTK

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

model.set_rope(RopeNTK(head_dim=64, base=10000.0,
                      train_seq_len=1024, target_seq_len=4096))
ntk_ppl = {}
for L in context_lengths:
    ntk_ppl[L] = evaluate_perplexity(model, eval_dataset, seq_len=L)
    print(f"NTK: length={L}, ppl={ntk_ppl[L]:.2f}")
