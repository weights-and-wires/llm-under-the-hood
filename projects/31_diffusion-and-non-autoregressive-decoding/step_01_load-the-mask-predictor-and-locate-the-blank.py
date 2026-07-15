"""
Project 31: Step 1 — Load the mask predictor and locate the blank

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL = "bert-base-uncased"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForMaskedLM.from_pretrained(MODEL, dtype=torch.float32).eval()

MASK = tok.mask_token_id      # the integer id of the [MASK] blank
print("mask token:", tok.mask_token, "id:", MASK)
