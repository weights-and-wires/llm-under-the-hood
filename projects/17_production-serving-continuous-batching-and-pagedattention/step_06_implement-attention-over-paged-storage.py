"""
Project 17: Step 6 — Implement attention over paged storage

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def paged_attention(
    query: Tensor,                   # (num_heads, head_dim)
    page_table: list[int],           # logical_block -> physical_page
    block_pool_storage: Tensor,      # (num_pages, PAGE_SIZE, num_heads, head_dim, 2)
    seq_len: int,                    # number of tokens stored
) -> Tensor:
    out = torch.zeros_like(query)
    scores = torch.zeros(seq_len, device=query.device)

    for logical_block, physical_page in enumerate(page_table):
        page_start = logical_block * PAGE_SIZE
        page_end = min(page_start + PAGE_SIZE, seq_len)
        for t in range(page_start, page_end):
            slot = t - page_start
            k = block_pool_storage[physical_page, slot, :, :, 0]
            scores[t] = (query * k).sum() / math.sqrt(head_dim)

    weights = torch.softmax(scores[:seq_len], dim=-1)

    for logical_block, physical_page in enumerate(page_table):
        page_start = logical_block * PAGE_SIZE
        page_end = min(page_start + PAGE_SIZE, seq_len)
        for t in range(page_start, page_end):
            slot = t - page_start
            v = block_pool_storage[physical_page, slot, :, :, 1]
            out += weights[t] * v

    return out
