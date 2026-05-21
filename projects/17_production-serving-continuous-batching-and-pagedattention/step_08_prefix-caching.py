"""
Project 17: Step 8 — Prefix caching

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class PrefixCache:
    def __init__(self):
        self.cache: dict[tuple[int, ...], list[int]] = {}
        self.refcount: dict[int, int] = defaultdict(int)

    def find_prefix_match(self, tokens: list[int]) -> tuple[int, list[int]]:
        for n in range(len(tokens) // PAGE_SIZE * PAGE_SIZE, 0, -PAGE_SIZE):
            key = tuple(tokens[:n])
            if key in self.cache:
                pages = self.cache[key]
                for p in pages:
                    self.refcount[p] += 1
                return n, pages
        return 0, []
