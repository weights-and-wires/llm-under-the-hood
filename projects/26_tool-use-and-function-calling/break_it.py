"""
Project 26: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

def corrupt(s: str) -> str:
    # Three random corruptions, applied with probability 1.0.
    s = s.replace("}", "")        # remove closing braces
    s = s.replace("\"", "'")       # swap double quotes for single
    s = s.replace(":", " is")      # replace colons with " is"
    return s

def dispatch(action: dict) -> str:
    # ... existing code ...
    raw = fn(**args)
    return corrupt(str(raw))
