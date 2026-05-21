"""
Project 28: Step 6 — Add a cross-encoder re-ranker

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def retrieve_and_rerank(query, encoder, reranker, index, corpus, k_recall=50, k_final=3):
    q_vec = encoder.encode([query])
    q_vec = q_vec / np.linalg.norm(q_vec)
    _, ids = index.search(q_vec.astype(np.float32), k_recall)
    candidates = [corpus[i] for i in ids[0]]
    pairs = [(query, doc) for doc in candidates]
    scores = reranker.predict(pairs)
    top_idx = np.argsort(-scores)[:k_final]
    return [candidates[i] for i in top_idx]
