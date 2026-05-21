"""
Project 14: Step 1 — Train a tiny draft model on the same data

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def measure_tokens_per_second(model, prompt_ids, num_tokens=128):
    model.eval()
    start = time.time()
    ids = prompt_ids.clone()
    with torch.no_grad():
        for _ in range(num_tokens):
            logits = model(ids)[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
    elapsed = time.time() - start
    return num_tokens / elapsed
