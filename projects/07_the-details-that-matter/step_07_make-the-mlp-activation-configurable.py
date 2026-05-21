"""
Project 7: Step 7 — Make the MLP activation configurable

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.d_model

        if config.activation == "swiglu":
            self.kind = "swiglu"
            self.w1 = nn.Linear(config.d_model, hidden, bias=False)
            self.w2 = nn.Linear(config.d_model, hidden, bias=False)
            self.w3 = nn.Linear(hidden, config.d_model, bias=False)
        else:
            self.kind = config.activation
            self.fc1 = nn.Linear(config.d_model, hidden)
            self.fc2 = nn.Linear(hidden, config.d_model)

    def forward(self, x):
        if self.kind == "gelu":
            return self.fc2(gelu(self.fc1(x)))
        if self.kind == "relu2":
            return self.fc2(torch.relu(self.fc1(x)) ** 2)
        if self.kind == "swiglu":
            return self.w3(F.silu(self.w1(x)) * self.w2(x))
        raise ValueError(self.kind)
