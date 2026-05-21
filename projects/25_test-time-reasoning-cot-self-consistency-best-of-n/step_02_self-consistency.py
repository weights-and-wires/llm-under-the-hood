"""
Project 25: Step 2 — Self-consistency

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def self_consistency(question, n=8, temperature=0.7):
    answers = []
    for _ in range(n):
        text = run_one(cot_prompt(question), temperature=temperature)
        ans = extract_answer(text)
        if ans is not None:
            answers.append(ans)
    if not answers:
        return None
    counts = Counter(answers)
    return counts.most_common(1)[0][0]
