"""
Project 21: Step 5 — Add LoRA instead of full fine-tuning

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)

        self.rank = rank
        self.scale = alpha / rank

        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]

        self.A = nn.Parameter(torch.zeros(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.A.t()) @ self.B.t()
        return base_out + self.scale * lora_out
