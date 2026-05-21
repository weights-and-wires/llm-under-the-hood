"""
Project 24: Step 7 — Implement SimPO

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def simpo_loss(policy_chosen_logps, policy_rejected_logps,
               chosen_lens, rejected_lens, beta=2.0, gamma=1.0):
    # Length-normalized log-probs
    chosen_avg_logps = policy_chosen_logps / chosen_lens
    rejected_avg_logps = policy_rejected_logps / rejected_lens

    # Target reward margin
    logits = beta * (chosen_avg_logps - rejected_avg_logps) - gamma

    loss = -F.logsigmoid(logits).mean()
    return loss
