"""
Project 13: Step 4 — Change attention to decode one token at a time

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def decode_step(x_new: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, pos: int) -> torch.Tensor:
    q_new = q_proj(x_new)
    k_new = k_proj(x_new)
    v_new = v_proj(x_new)

    q_new = split_heads(q_new)  # [B, H, 1, d]
    k_new = split_heads(k_new)  # [B, H, 1, d]
    v_new = split_heads(v_new)  # [B, H, 1, d]

    k_cache[:, :, pos:pos+1, :] = k_new
    v_cache[:, :, pos:pos+1, :] = v_new

    k_all = k_cache[:, :, :pos+1, :]
    v_all = v_cache[:, :, :pos+1, :]

    scores = q_new @ k_all.transpose(-2, -1) / math.sqrt(q_new.size(-1))
    att = scores.softmax(dim=-1)
    out = att @ v_all
    return merge_heads(out)
