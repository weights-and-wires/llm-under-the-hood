"""
Project 8: Step 2 — Decompose the computation into tiles

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

BR = 64   # query row tile size
BC = 64   # key/value column tile size

def tiled_attention_skeleton(q, k, v, BR=64, BC=64):
    B, H, N, D = q.shape
    out = torch.zeros_like(q)

    for i in range(0, N, BR):
        q_tile = q[:, :, i:i+BR, :]                   # (B, H, BR, D)
        # running statistics for this query row tile, set up in Step 3
        for j in range(0, N, BC):
            k_tile = k[:, :, j:j+BC, :]               # (B, H, BC, D)
            v_tile = v[:, :, j:j+BC, :]               # (B, H, BC, D)
            scores_tile = q_tile @ k_tile.transpose(-2, -1)  # (B, H, BR, BC)
            # update running statistics and partial output using scores_tile
            ...
        # write final row tile of out
        out[:, :, i:i+BR, :] = ...
    return out
