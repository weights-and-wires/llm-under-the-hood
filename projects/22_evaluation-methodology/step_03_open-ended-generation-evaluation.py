"""
Project 22: Step 3 — Open-ended generation evaluation

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

@torch.no_grad()
def generate_eval(model, tokenizer, prompt, reference, max_new_tokens=64):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    out = model.generate(ids, max_new_tokens=max_new_tokens,
                         do_sample=False, temperature=0.0)
    completion = tokenizer.decode(out[0][ids.shape[1]:],
                                  skip_special_tokens=True)
    exact = completion.strip() == reference.strip()
    sub = reference.strip() in completion
    return completion, exact, sub
