"""
Project 22: Step 7 — The held-out paraphrased variant

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

PARAPHRASE_PROMPT = """Rewrite the following question so that it asks
exactly the same thing, but uses different words. Do not change the
meaning. Do not add or remove information.

Original: {question}

Rewrite:"""

def paraphrase_eval(client, questions):
    paraphrased = []
    for q in questions:
        resp = client.complete(PARAPHRASE_PROMPT.format(question=q),
                               max_tokens=256, temperature=0.7)
        paraphrased.append(resp.strip())
    return paraphrased
