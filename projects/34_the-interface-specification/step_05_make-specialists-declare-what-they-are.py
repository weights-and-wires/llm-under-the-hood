"""
Project 34: Step 5 — Make specialists declare what they are

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

ckpt["interface"] = {
    "interface_name": "COMPOSE-LLM-IFACE",
    "interface_version": "1.0",
    "base_model_id": "trunk-init-2026-03",
    "shared_prefix_depth": 8,
    "d_model": 768,
    "n_heads": 12,
    "norm_type": "RMSNorm",
    "norm_eps": 1e-5,
    "positional_encoding": "RoPE",
    "rope_theta": 10000,
    "tokenizer_hash": "sha256:..."
}
