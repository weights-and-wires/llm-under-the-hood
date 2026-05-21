"""
Project 17: Step 4 — Implement the paged KV cache

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

PAGE_SIZE = 16  # tokens per page

class BlockPool:
    def __init__(self, num_pages, page_shape, dtype, device):
        # one big tensor for all pages, indexed by physical page id
        self.storage = torch.zeros(
            (num_pages, *page_shape), dtype=dtype, device=device,
        )
        self.free_list: list[int] = list(range(num_pages))

    def alloc(self) -> int:
        if not self.free_list:
            raise OutOfPagesError()
        return self.free_list.pop()

    def free(self, page_id: int):
        self.free_list.append(page_id)

class PagedKVCache:
    def __init__(self, pool: BlockPool):
        self.pool = pool
        self.page_table: list[int] = []     # logical block -> physical page
        self.length: int = 0                # number of tokens stored

    def append(self, k: Tensor, v: Tensor):
        # k, v shapes: (num_layers, num_kv_heads, head_dim)
        slot_in_page = self.length % PAGE_SIZE
        if slot_in_page == 0:
            self.page_table.append(self.pool.alloc())
        page_id = self.page_table[-1]
        self.pool.storage[page_id, slot_in_page, :, :, 0] = k
        self.pool.storage[page_id, slot_in_page, :, :, 1] = v
        self.length += 1

    def free_all(self):
        for page_id in self.page_table:
            self.pool.free(page_id)
        self.page_table = []
        self.length = 0
