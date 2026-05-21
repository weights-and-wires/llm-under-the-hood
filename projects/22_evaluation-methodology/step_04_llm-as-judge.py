"""
Project 22: Step 4 — LLM-as-judge

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

JUDGE_PROMPT = """You are grading an answer to a question.

Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}

Rate the candidate answer from 1 to 5 on correctness and clarity.
Output only the number, no other text.

Score:"""

def llm_judge(client, question, reference, candidate):
    prompt = JUDGE_PROMPT.format(question=question,
                                 reference=reference,
                                 candidate=candidate)
    resp = client.complete(prompt, max_tokens=4, temperature=0.0)
    try:
        return int(resp.strip())
    except ValueError:
        return None
