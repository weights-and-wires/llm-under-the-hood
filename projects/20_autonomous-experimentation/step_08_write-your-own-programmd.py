"""
Project 20: Step 8 — Write your own `program.md`

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

if new_val_bpb < best_val_bpb:
    keep_change()
else:
    discard_change()

if new_memory_gb < best_memory_gb and new_val_bpb <= baseline_val_bpb + 0.005:
    keep_change()
else:
    discard_change()
