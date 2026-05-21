"""
Project 25: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

def truncate_chain(chain):
    steps = chain.split("\n")
    keep = max(1, len(steps) // 2)
    return "\n".join(steps[:keep])

broken_data = [(q, truncate_chain(c), rollout_label(q, truncate_chain(c)))
               for q, c, _ in original_data]
