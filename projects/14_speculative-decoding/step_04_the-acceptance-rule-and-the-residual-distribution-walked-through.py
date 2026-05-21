"""
Project 14: Step 4 — The acceptance rule and the residual distribution, walked through

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def speculative_step(draft_model, main_model, ids, K=4):
    candidate_ids, draft_dists = draft_propose(draft_model, ids, K)
    main_dists, bonus_dist = main_verify(main_model, ids, candidate_ids)

    accepted_ids = []
    for i in range(K):
        t = candidate_ids[:, i:i+1]
        q_full, _ = draft_dists[i]
        p_full = main_dists[i]
        q_t = q_full.gather(-1, t)
        p_t = p_full.gather(-1, t)
        u = torch.rand_like(q_t)
        if (u < torch.clamp(p_t / q_t, max=1.0)).all():
            accepted_ids.append(t)
        else:
            # Residual distribution: max(0, p - q), normalized.
            residual = torch.clamp(p_full - q_full, min=0.0)
            residual = residual / residual.sum(dim=-1, keepdim=True)
            replacement = torch.multinomial(residual, num_samples=1)
            accepted_ids.append(replacement)
            return torch.cat(accepted_ids, dim=1)

    # All K accepted: append the bonus token from the main model.
    bonus = torch.multinomial(bonus_dist, num_samples=1)
    accepted_ids.append(bonus)
    return torch.cat(accepted_ids, dim=1)
