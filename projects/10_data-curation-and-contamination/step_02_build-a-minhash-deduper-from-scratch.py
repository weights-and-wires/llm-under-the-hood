"""
Project 10: Step 2 — Build a MinHash deduper from scratch

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import mmh3

def shingles(text: str, k: int = 5) -> set:
    tokens = text.split()
    if len(tokens) < k:
        return set()
    return {mmh3.hash64(" ".join(tokens[i:i+k]))[0]
            for i in range(len(tokens) - k + 1)}

import numpy as np

def minhash_signature(shingle_set: set, num_hashes: int = 128,
                      seed: int = 42) -> np.ndarray:
    if not shingle_set:
        return np.full(num_hashes, np.iinfo(np.uint64).max,
                       dtype=np.uint64)
    rng = np.random.default_rng(seed)
    # K independent hash functions implemented as (a*x + b) mod p
    p = (1 << 61) - 1
    a = rng.integers(1, p, size=num_hashes, dtype=np.uint64)
    b = rng.integers(0, p, size=num_hashes, dtype=np.uint64)
    shingles_arr = np.fromiter(shingle_set, dtype=np.uint64)
    # For each hash function, take the min over all shingles
    hashed = (np.outer(a, shingles_arr) + b[:, None]) % p
    return hashed.min(axis=1)

def lsh_buckets(signature: np.ndarray, num_bands: int = 32) -> list:
    bands = signature.reshape(num_bands, -1)
    return [mmh3.hash64(b.tobytes())[0] for b in bands]
