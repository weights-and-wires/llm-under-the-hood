"""
Project 10: Step 3 — Compute 13-gram contamination against an eval set

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def thirteen_grams(text: str) -> list:
    tokens = text.split()
    return [tuple(tokens[i:i+13]) for i in range(len(tokens) - 12)]

eval_index = {}
for eval_id, example in enumerate(eval_set):
    for gram in thirteen_grams(example["text"]):
        h = mmh3.hash64(" ".join(gram))[0]
        eval_index.setdefault(h, []).append(eval_id)

contaminated_docs = {}
for doc_id, doc in enumerate(training_corpus):
    hits = set()
    for gram in thirteen_grams(doc["text"]):
        h = mmh3.hash64(" ".join(gram))[0]
        if h in eval_index:
            hits.update(eval_index[h])
    if hits:
        contaminated_docs[doc_id] = hits
