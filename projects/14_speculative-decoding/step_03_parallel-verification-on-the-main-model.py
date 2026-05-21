"""
Project 14: Step 3 — Parallel verification on the main model

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def main_verify(main_model, ids, candidate_ids):
    full_seq = torch.cat([ids, candidate_ids], dim=1)
    with torch.no_grad():
        logits = main_model(full_seq)
    K = candidate_ids.shape[1]
    # We want the main model's distribution at the K positions where
    # the draft proposed tokens. If candidate token t_i was proposed
    # at position len(ids)+i, then the main model's distribution for
    # that position is at logits[:, len(ids)+i-1, :], because the
    # output at position j predicts the token at position j+1.
    main_dists = []
    start = ids.shape[1] - 1
    for i in range(K):
        p_logits = logits[:, start + i, :]
        main_dists.append(torch.softmax(p_logits, dim=-1))
    # The +1 distribution (after the last candidate) is the bonus
    # distribution we sample from if all K candidates are accepted.
    bonus_dist = torch.softmax(logits[:, start + K, :], dim=-1)
    return main_dists, bonus_dist
