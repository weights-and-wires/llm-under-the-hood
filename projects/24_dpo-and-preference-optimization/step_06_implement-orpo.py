"""
Project 24: Step 6 — Implement ORPO

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def orpo_loss(policy_chosen_logps, policy_rejected_logps, chosen_labels, lambda_=0.1):
    # SFT term: standard negative log-likelihood on chosen
    nll_loss = -policy_chosen_logps.mean()

    # Odds ratio term
    log_odds = (policy_chosen_logps - torch.log(1 - torch.exp(policy_chosen_logps))) \
             - (policy_rejected_logps - torch.log(1 - torch.exp(policy_rejected_logps)))
    or_loss = -F.logsigmoid(log_odds).mean()

    return nll_loss + lambda_ * or_loss
