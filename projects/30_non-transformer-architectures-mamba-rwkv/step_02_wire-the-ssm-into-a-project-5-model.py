"""
Project 30: Step 2 — Wire the SSM into a Project-5 model

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class SSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_ff: int = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.RMSNorm(d_model)
        self.ssm = S6Block(d_model, d_state)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
