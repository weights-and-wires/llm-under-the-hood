"""
Project 16: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

# Disable extension
model.set_rope(RopeRaw(head_dim=64, base=10000.0,
                      max_seq_len=4096))

break_lengths = [1126, 1228, 1536, 2048]
break_ppl = {}
for L in break_lengths:
    break_ppl[L] = evaluate_perplexity(model, eval_dataset, seq_len=L)
    print(f"RAW: length={L}, ppl={break_ppl[L]:.2f}")
