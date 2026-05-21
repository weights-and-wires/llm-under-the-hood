"""
Project 29: Step 5 — Instruction-tune on visual question answering

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

VQA_TEMPLATES = {
    "dog": [
        ("What animal is in the picture?", "A dog."),
        ("Is this a cat or a dog?", "A dog."),
        ("Describe what you see.", "I see a dog."),
    ],
    # ... and so on
}
