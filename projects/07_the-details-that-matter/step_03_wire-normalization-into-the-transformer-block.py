"""
Project 7: Step 3 — Wire normalization into the transformer block

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def make_norm(norm_type: str, d_model: int) -> nn.Module:
    if norm_type == "layernorm":
        return LayerNorm(d_model)
    if norm_type == "rmsnorm":
        return RMSNorm(d_model)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm_type: {norm_type}")

self.norm1 = make_norm(config.norm_type, config.d_model)
self.norm2 = make_norm(config.norm_type, config.d_model)

x = x + self.attn(self.norm1(x))
x = x + self.mlp(self.norm2(x))
