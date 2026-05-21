"""
Project 22: Step 2 — Multiple-choice scoring (MMLU-style)

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

@torch.no_grad()
def mmlu_score(model, tokenizer, question, options, answer_label):
    prompt = (f"Question: {question}\n"
              f"A. {options[0]}\n"
              f"B. {options[1]}\n"
              f"C. {options[2]}\n"
              f"D. {options[3]}\n"
              f"Answer:")
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    logits = model(ids).logits[0, -1]
    log_probs = F.log_softmax(logits, dim=-1)
    letter_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[0]
                  for c in "ABCD"]
    scores = [log_probs[i].item() for i in letter_ids]
    pick = "ABCD"[max(range(4), key=lambda i: scores[i])]
    return pick, pick == answer_label, scores
