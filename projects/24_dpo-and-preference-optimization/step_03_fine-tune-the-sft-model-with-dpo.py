"""
Project 24: Step 3 — Fine-tune the SFT model with DPO

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

for batch in train_loader:
    prompt_ids, chosen_ids, rejected_ids = batch

    # Forward pass through policy (trains)
    policy_chosen_logps = compute_logprobs(policy, prompt_ids, chosen_ids)
    policy_rejected_logps = compute_logprobs(policy, prompt_ids, rejected_ids)

    # Forward pass through reference (no grad)
    with torch.no_grad():
        ref_chosen_logps = compute_logprobs(ref_model, prompt_ids, chosen_ids)
        ref_rejected_logps = compute_logprobs(ref_model, prompt_ids, rejected_ids)

    loss = dpo_loss(policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps, beta=0.1)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def compute_logprobs(model, prompt_ids, completion_ids):
    full_ids = torch.cat([prompt_ids, completion_ids], dim=-1)
    logits = model(full_ids).logits[:, :-1]  # next-token logits
    targets = full_ids[:, 1:]
    logprobs = F.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # Mask out prompt tokens
    completion_mask = torch.zeros_like(targets, dtype=torch.bool)
    completion_mask[:, prompt_ids.size(-1) - 1:] = True
    return (token_logprobs * completion_mask).sum(dim=-1)
