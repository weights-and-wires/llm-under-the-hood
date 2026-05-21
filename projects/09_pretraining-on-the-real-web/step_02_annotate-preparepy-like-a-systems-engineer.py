"""
Project 9: Step 2 — Annotate `prepare.py` like a systems engineer

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

# Input: dataset rows with raw text and metadata
# Operation: keep only the text field
# Output: plain text documents
# Why here: avoids carrying useless metadata into tokenization and storage

# Input: one text document
# Operation: tokenize into integer IDs and append end-of-document token
# Output: list of token IDs for one document
# Why here: preserves document boundaries and makes training data
#   ready for direct loading

# Input: many tokenized documents
# Operation: pack tokens into shard-sized arrays and write to disk
# Output: shard files containing integer token IDs
# Why here: training can stream tokens directly without repeated preprocessing
