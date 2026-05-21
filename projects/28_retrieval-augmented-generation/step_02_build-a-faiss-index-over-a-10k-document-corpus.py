"""
Project 28: Step 2 — Build a FAISS index over a 10K-document corpus

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import faiss
import numpy as np

doc_vectors = encode_documents(corpus)
doc_vectors = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)
index = faiss.IndexFlatIP(doc_vectors.shape[1])
index.add(doc_vectors.astype(np.float32))
