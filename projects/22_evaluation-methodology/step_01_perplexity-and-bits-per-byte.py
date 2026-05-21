"""
Project 22: Step 1 — Perplexity and bits per byte

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def perplexity_and_bpb(model, tokenizer, text):
    enc = tokenizer.encode(text, return_tensors="pt").to(model.device)
    n_bytes = len(text.encode("utf-8"))
    logits = model(enc).logits[:, :-1, :]
    targets = enc[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    chosen = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    total_nats = -chosen.sum().item()
    n_tokens = targets.numel()
    ppl = math.exp(total_nats / n_tokens)
    bpb = total_nats / math.log(2) / n_bytes
    return ppl, bpb
