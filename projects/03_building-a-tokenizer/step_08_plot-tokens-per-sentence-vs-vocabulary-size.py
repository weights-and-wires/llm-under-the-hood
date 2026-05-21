"""
Project 3: Step 8 — Plot tokens-per-sentence vs vocabulary size

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

sizes = [256, 1024, 4096, 8192]
results = []

for vocab_size in sizes:
    tokenizer = train_bpe(corpus_text, vocab_size)
    token_counts = []
    for sentence in sample_sentences:
        ids = tokenizer.encode(sentence)
        token_counts.append(len(ids))
    avg_tokens = sum(token_counts) / len(token_counts)
    results.append((vocab_size, avg_tokens))
