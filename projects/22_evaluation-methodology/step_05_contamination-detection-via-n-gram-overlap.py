"""
Project 22: Step 5 — Contamination detection via n-gram overlap

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def ngrams(tokens, n=13):
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

def build_eval_index(eval_texts, tokenizer, n=13):
    index = set()
    for text in eval_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        index.update(ngrams(ids, n))
    return index

def scan_corpus(corpus_iter, eval_index, tokenizer, n=13):
    hits = []
    for doc_id, doc in corpus_iter:
        ids = tokenizer.encode(doc, add_special_tokens=False)
        doc_ngrams = ngrams(ids, n)
        overlap = doc_ngrams & eval_index
        if overlap:
            hits.append((doc_id, len(overlap)))
    return hits
