"""
Project 24: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

# Original:
# policy_chosen_logps = compute_logprobs(policy, prompt_ids, chosen_ids)
# policy_rejected_logps = compute_logprobs(policy, prompt_ids, rejected_ids)

# Swapped:
policy_chosen_logps = compute_logprobs(policy, prompt_ids, rejected_ids)
policy_rejected_logps = compute_logprobs(policy, prompt_ids, chosen_ids)

with torch.no_grad():
    ref_chosen_logps = compute_logprobs(ref_model, prompt_ids, rejected_ids)
    ref_rejected_logps = compute_logprobs(ref_model, prompt_ids, chosen_ids)
