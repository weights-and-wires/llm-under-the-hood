"""
Project 30: Step 5 — Train RWKV on the same data

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class RWKVBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.tm = RWKVTimeMix(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.cm = ChannelMix(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.tm(self.norm1(x))
        x = x + self.cm(self.norm2(x))
        return x
