"""
Project 25: Step 3 — Train a tiny outcome reward model

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class ORM(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.head = nn.Linear(base_model.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.base(input_ids, attention_mask=attention_mask).last_hidden_state
        last = h[torch.arange(h.size(0)), attention_mask.sum(dim=1) - 1]
        return self.head(last).squeeze(-1)
