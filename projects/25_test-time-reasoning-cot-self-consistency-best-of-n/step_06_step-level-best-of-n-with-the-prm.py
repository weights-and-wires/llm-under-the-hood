"""
Project 25: Step 6 — Step-level Best-of-N with the PRM

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def step_level_bon(question, k=4, max_steps=12):
    chain = ""
    for _ in range(max_steps):
        candidates = []
        for _ in range(k):
            step = sample_step(question, chain, temperature=0.7)
            score = prm.score(question, chain + step)
            candidates.append((score, step))
        candidates.sort(key=lambda x: -x[0])
        best_step = candidates[0][1]
        chain += best_step
        if has_final_answer(chain):
            break
    return extract_answer(chain)
