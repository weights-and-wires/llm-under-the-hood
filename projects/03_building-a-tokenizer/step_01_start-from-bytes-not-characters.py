"""
Project 3: Step 1 — Start from bytes, not characters

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

text = "hello 👋"
data = text.encode("utf-8")
print(list(data))
# [104, 101, 108, 108, 111, 32, 240, 159, 145, 139]
