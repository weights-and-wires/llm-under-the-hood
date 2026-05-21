"""
Project 14: Step 2 — Naive draft generation of K=4 tokens

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def draft_propose(draft_model, ids, K=4):
    candidate_ids = []
    candidate_qs = []
    cur = ids.clone()
    for _ in range(K):
        with torch.no_grad():
            logits = draft_model(cur)[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            q = probs.gather(-1, next_id)
        candidate_ids.append(next_id)
        candidate_qs.append((probs, next_id))  # store full dist for residual
        cur = torch.cat([cur, next_id], dim=1)
    return torch.cat(candidate_ids, dim=1), candidate_qs
