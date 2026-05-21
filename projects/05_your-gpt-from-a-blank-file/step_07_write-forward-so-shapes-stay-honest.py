"""
Project 5: Step 7 — Write `forward()` so shapes stay honest

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple:
    B, T = idx.shape
    assert T <= self.cfg.block_size

    pos = torch.arange(0, T, device=idx.device)
    x = self.token_embedding(idx) + self.position_embedding(pos)

    for block in self.blocks:
        x = block(x)

    x = self.ln_f(x)
    logits = self.lm_head(x)

    loss = None
    if targets is not None:
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

    return logits, loss
