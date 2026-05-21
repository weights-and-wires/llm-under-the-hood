"""
Project 14: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

mis_draft = train_draft_on_different_data(...)
tps_aligned, acc_aligned = measure_speculative_tps(
    aligned_draft, main, prompt, K=4
)
tps_mismatched, acc_mismatched = measure_speculative_tps(
    mis_draft, main, prompt, K=4
)
baseline_tps = measure_tokens_per_second(main, prompt)

print(f"Baseline: {baseline_tps:.1f} tps")
print(f"Aligned spec: {tps_aligned:.1f} tps "
      f"(α={acc_aligned:.2f}, {tps_aligned/baseline_tps:.2f}x)")
print(f"Mismatched spec: {tps_mismatched:.1f} tps "
      f"(α={acc_mismatched:.2f}, {tps_mismatched/baseline_tps:.2f}x)")
