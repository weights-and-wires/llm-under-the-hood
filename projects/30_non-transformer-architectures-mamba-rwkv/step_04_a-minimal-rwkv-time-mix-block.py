"""
Project 30: Step 4 — A minimal RWKV time-mix block

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class RWKVTimeMix(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Per-channel decay rate, kept positive via softplus.
        self.w_raw = nn.Parameter(torch.zeros(d_model))
        # Per-channel projections.
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        w = torch.nn.functional.softplus(self.w_raw)   # (D,)
        decay = torch.exp(-w)                          # (D,)

        k = self.key(x)        # (B, L, D)
        v = self.value(x)      # (B, L, D)
        r = torch.sigmoid(self.receptance(x))  # (B, L, D)

        num = torch.zeros(B, D, device=x.device)
        den = torch.zeros(B, D, device=x.device) + 1e-8
        outs = []
        for t in range(L):
            kt = k[:, t]
            vt = v[:, t]
            num = decay * num + torch.exp(kt) * vt
            den = decay * den + torch.exp(kt)
            wkv = num / den
            outs.append(r[:, t] * wkv)
        out = torch.stack(outs, dim=1)        # (B, L, D)
        return self.output(out)

class ChannelMix(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.key = nn.Linear(d_model, d_ff, bias=False)
        self.value = nn.Linear(d_ff, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.relu(self.key(x)) ** 2     # squared ReLU
        v = self.value(k)
        r = torch.sigmoid(self.receptance(x))
        return r * v
