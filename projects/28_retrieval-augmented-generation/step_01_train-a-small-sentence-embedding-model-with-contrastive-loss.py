"""
Project 28: Step 1 — Train a small sentence-embedding model with contrastive loss

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def info_nce_loss(query_embs, passage_embs, tau=0.07):
    sim = query_embs @ passage_embs.T / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)
