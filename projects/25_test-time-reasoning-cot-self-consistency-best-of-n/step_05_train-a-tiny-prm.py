"""
Project 25: Step 5 — Train a tiny PRM

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def rollout_label(question, prefix, m=8):
    correct = 0
    for _ in range(m):
        continuation = run_one(prefix, temperature=0.7)
        ans = extract_answer(prefix + continuation)
        if ans == ground_truth[question]:
            correct += 1
    return correct / m
