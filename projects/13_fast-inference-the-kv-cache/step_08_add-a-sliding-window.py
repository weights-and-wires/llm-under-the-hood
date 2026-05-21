"""
Project 13: Step 8 — Add a sliding window

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

if cache_len > window:
    k_cache = k_cache[:, :, -window:, :]
    v_cache = v_cache[:, :, -window:, :]
