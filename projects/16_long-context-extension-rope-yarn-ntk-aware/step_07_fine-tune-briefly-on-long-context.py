"""
Project 16: Step 7 — Fine-tune briefly on long context

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

# Apply YaRN rotation
model.set_rope(RopeYaRN(...))

# Long-context fine-tune
long_dataset = LongContextDataset(tokenizer,
                                  files=long_docs,
                                  seq_len=4096)
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=2e-5)

for step in range(1000):
    batch = next(long_dataset)
    loss = model(batch).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    if step % 50 == 0:
        print(f"step {step}, loss {loss.item():.3f}")
