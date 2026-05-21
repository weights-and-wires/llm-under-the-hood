"""
Project 19: Step 8 — Build the cost calculator

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def train_time_hours(params: int, tokens_per_second: float, tokens_per_param: int = 8) -> float:
    target_tokens = tokens_per_param * params
    seconds = target_tokens / tokens_per_second
    return seconds / 3600

print(train_time_hours(params=200_000_000, tokens_per_second=40_000))
