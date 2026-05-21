"""
Project 8: Step 4 — Verify against the naive version

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

torch.manual_seed(0)
B, H, N, D = 2, 4, 256, 32
q = torch.randn(B, H, N, D, device="cuda")
k = torch.randn(B, H, N, D, device="cuda")
v = torch.randn(B, H, N, D, device="cuda")

ref = naive_attention(q, k, v)
out = tiled_attention(q, k, v, BR=32, BC=32)

diff = (ref - out).abs().max().item()
print(f"max absolute difference: {diff:.6e}")
