"""
Project 25: Step 4 — Best-of-N with the ORM

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def bon(question, n=8, temperature=0.7):
    candidates = []
    for _ in range(n):
        text = run_one(cot_prompt(question), temperature=temperature)
        score = orm.score(question, text)
        candidates.append((score, text))
    candidates.sort(key=lambda x: -x[0])
    best = candidates[0][1]
    return extract_answer(best)
