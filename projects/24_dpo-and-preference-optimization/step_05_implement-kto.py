"""
Project 24: Step 5 — Implement KTO

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def kto_loss(policy_logps, ref_logps, label, beta=0.1, desirable_weight=1.0, undesirable_weight=1.0):
    # Implicit reward
    logratio = policy_logps - ref_logps

    # Baseline: KL divergence estimate from a mismatched batch
    # (computed elsewhere; here we just use the mean as a stand-in)
    kl_baseline = compute_kl_baseline(policy, ref_model)

    if label == "desirable":
        loss = desirable_weight * (1 - torch.sigmoid(beta * (logratio - kl_baseline)))
    else:  # undesirable
        loss = undesirable_weight * (1 - torch.sigmoid(beta * (kl_baseline - logratio)))

    return loss.mean()
