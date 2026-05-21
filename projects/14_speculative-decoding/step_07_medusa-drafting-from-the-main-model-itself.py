"""
Project 14: Step 7 — Medusa — drafting from the main model itself

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class MedusaHead(nn.Module):
    def __init__(self, d_model, vocab_size, depth=1):
        super().__init__()
        # A small residual MLP on top of the main hidden state.
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(d_model, vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h)
