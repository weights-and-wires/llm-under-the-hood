"""
Project 23: Step 10 — Compare Step 0 and Step 10 Outputs by Hand

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

for prompts in dataloader:
    candidates = [policy.generate(prompts) for _ in range(G)]  # G samples per prompt
    rewards = reward_model.score(prompts, candidates)

    advantages = rewards - rewards.mean(dim=1, keepdim=True)

    logps = policy.logprob(prompts, candidates)

    loss = -(advantages.detach() * logps).mean()
    loss = loss + beta * kl_to_reference(policy, ref_policy, prompts)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
