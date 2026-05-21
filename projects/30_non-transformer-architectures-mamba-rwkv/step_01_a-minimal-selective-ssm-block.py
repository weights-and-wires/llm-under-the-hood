"""
Project 30: Step 1 — A minimal selective SSM block

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import torch
import torch.nn as nn

class S6Block(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A is stored as log(-A) so A stays negative when exponentiated.
        # Negative real A means the recurrence is stable.
        A = torch.arange(1, d_state + 1).float().expand(d_model, d_state)
        self.A_log = nn.Parameter(torch.log(A))

        # Input-dependent projections: B, C, and the step size delta
        # are all produced from x by a linear layer.
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)

        # The step size delta also gets a learned bias before softplus.
        self.dt_proj = nn.Linear(1, d_model)

        # Skip-connection multiplier (per channel).
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = self.d_state

        # Project x into B, C, and delta.
        x_proj = self.x_proj(x)              # (B, L, 2N + 1)
        B_in, C_in, dt = x_proj.split(
            [N, N, 1], dim=-1)
        dt = torch.nn.functional.softplus(
            self.dt_proj(dt))                # (B, L, D)

        A = -torch.exp(self.A_log)           # (D, N), all negative

        # Discretize: A_bar = exp(dt * A), B_bar approximation
        # collapses to dt * B for small dt.
        A_bar = torch.exp(dt.unsqueeze(-1) * A)   # (B, L, D, N)
        B_bar = dt.unsqueeze(-1) * B_in.unsqueeze(-2)  # (B, L, D, N)

        # Recurrence: walk through the sequence one step at a time.
        s = torch.zeros(B, D, N, device=x.device)
        ys = []
        for t in range(L):
            s = A_bar[:, t] * s + B_bar[:, t] * x[:, t].unsqueeze(-1)
            y = (C_in[:, t].unsqueeze(1) * s).sum(-1)   # (B, D)
            ys.append(y)
        y = torch.stack(ys, dim=1)            # (B, L, D)

        # The D parameter is a residual skip: a direct linear path
        # from input to output, in parallel with the SSM.
        return y + self.D * x
