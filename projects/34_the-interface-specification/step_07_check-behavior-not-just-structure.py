"""
Project 34: Step 7 — Check behavior, not just structure

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

with torch.no_grad():
    x = reference_hidden_state  # fixed saved tensor [B, T, 768]
    y = specialist(x)

    if torch.isnan(y).any():
        raise ValueError("Dynamic check failed: NaNs in output")

    rms = y.pow(2).mean().sqrt().item()
    if not (0.1 < rms < 10.0):
        raise ValueError(f"Dynamic check failed: output RMS out of range: {rms:.4f}")
