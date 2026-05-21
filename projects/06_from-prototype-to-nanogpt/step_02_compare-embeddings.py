"""
Project 6: Step 2 — Compare embeddings

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

tok_emb = self.token_embedding(idx)      # (B, T, d)
pos = torch.arange(T, device=idx.device)
pos_emb = self.position_embedding(pos)   # (T, d)
x = tok_emb + pos_emb
