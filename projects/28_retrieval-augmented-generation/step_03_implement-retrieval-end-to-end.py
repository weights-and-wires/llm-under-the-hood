"""
Project 28: Step 3 — Implement retrieval end-to-end

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def retrieve(query, encoder, index, corpus, k=5):
    q_vec = encoder.encode([query])
    q_vec = q_vec / np.linalg.norm(q_vec)
    scores, ids = index.search(q_vec.astype(np.float32), k)
    return [(corpus[i], float(s)) for i, s in zip(ids[0], scores[0])]
