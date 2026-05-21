"""
Project 31: Step 6 — Fine-tune each freeze configuration under matched conditions

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
