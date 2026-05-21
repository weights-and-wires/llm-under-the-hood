"""
Project 17: Step 1 — Build the naive padded batcher

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

# inputs shape: (batch_size, max_prompt_len)
# kv_cache shape: (batch_size, max_seq_len, num_layers, num_heads, head_dim)

def naive_step(model, batch):
    # batch.tokens has shape (B, T), padded with PAD_TOKEN
    # batch.attention_mask has shape (B, T), 1 for real tokens, 0 for pad
    logits, batch.kv_cache = model(
        tokens=batch.tokens,
        attention_mask=batch.attention_mask,
        kv_cache=batch.kv_cache,
    )
    next_tokens = logits[:, -1, :].argmax(dim=-1)
    batch.tokens = torch.cat([batch.tokens, next_tokens.unsqueeze(-1)], dim=-1)
    batch.attention_mask = torch.cat(
        [batch.attention_mask, torch.ones_like(next_tokens).unsqueeze(-1)],
        dim=-1,
    )
    return batch
