"""
Project 18: Step 5 — Convert scores into routing probabilities

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

router_logits = self.router(x)           # (B, T, N)
router_probs = torch.softmax(router_logits, dim=-1)
