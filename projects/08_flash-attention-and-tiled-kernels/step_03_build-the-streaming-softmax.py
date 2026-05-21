"""
Project 8: Step 3 — Build the streaming softmax

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

m_tile = s_tile.max(dim=-1, keepdim=True).values     # local max of the tile
p_tile = torch.exp(s_tile - m_tile)                  # local stable exponentials
l_tile = p_tile.sum(dim=-1, keepdim=True)            # local sum of exponentials

m_new = torch.maximum(m_running, m_tile)
alpha = torch.exp(m_running - m_new)                 # rescaling for old stats
beta  = torch.exp(m_tile    - m_new)                 # rescaling for tile stats
l_new = alpha * l_running + beta * l_tile
o_new = alpha * o_running + beta * (p_tile @ v_tile)
m_running, l_running, o_running = m_new, l_new, o_new

o_final = o_running / l_running

def tiled_attention(q, k, v, BR=64, BC=64, causal=False):
    B, H, N, D = q.shape
    scale = D ** -0.5
    out = torch.zeros_like(q)

    for i in range(0, N, BR):
        q_tile = q[:, :, i:i+BR, :]
        m = torch.full((B, H, q_tile.shape[2], 1), float("-inf"),
                       device=q.device, dtype=q.dtype)
        l = torch.zeros((B, H, q_tile.shape[2], 1),
                        device=q.device, dtype=q.dtype)
        o = torch.zeros_like(q_tile)

        for j in range(0, N, BC):
            k_tile = k[:, :, j:j+BC, :]
            v_tile = v[:, :, j:j+BC, :]

            s = (q_tile @ k_tile.transpose(-2, -1)) * scale
            if causal:
                # mask future positions inside this tile
                row_idx = torch.arange(i, i + q_tile.shape[2], device=q.device)[:, None]
                col_idx = torch.arange(j, j + k_tile.shape[2], device=q.device)[None, :]
                s = s.masked_fill(col_idx > row_idx, float("-inf"))

            m_tile = s.max(dim=-1, keepdim=True).values
            p = torch.exp(s - m_tile)
            l_tile = p.sum(dim=-1, keepdim=True)

            m_new = torch.maximum(m, m_tile)
            alpha = torch.exp(m - m_new)
            beta = torch.exp(m_tile - m_new)
            l = alpha * l + beta * l_tile
            o = alpha * o + beta * (p @ v_tile)
            m = m_new

        out[:, :, i:i+BR, :] = o / l

    return out
