"""
Project 10: Step 4 — Build a mixing weight schedule across three domains

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def mixing_weights(step: int, total_steps: int) -> dict:
    progress = step / total_steps
    if progress < 0.8:
        return {"web": 0.85, "code": 0.10, "math": 0.05}
    # Ramp up code and math in the final 20%
    t = (progress - 0.8) / 0.2
    web = 0.85 - 0.30 * t
    code = 0.10 + 0.20 * t
    math = 0.05 + 0.10 * t
    return {"web": web, "code": code, "math": math}
