"""
Project 17: Step 2 — Build the continuous batcher

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

@dataclass
class Request:
    request_id: int
    prompt_tokens: list[int]
    output_tokens: list[int]
    kv_cache: KVCacheHandle      # see Step 4 for paged version
    max_tokens: int
    finished: bool = False

class Scheduler:
    def __init__(self, model, max_batch_size):
        self.model = model
        self.max_batch_size = max_batch_size
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def step(self):
        # 1. Retire finished requests
        self.running = [r for r in self.running if not r.finished]

        # 2. Admit new requests up to the batch capacity
        while len(self.running) < self.max_batch_size and self.waiting:
            new_req = self.waiting.popleft()
            self._prefill(new_req)
            self.running.append(new_req)

        # 3. One decode step over the running set
        if not self.running:
            return
        logits = self.model.decode_batch(self.running)
        next_tokens = logits.argmax(dim=-1)

        # 4. Append the new token to each request
        for req, tok in zip(self.running, next_tokens.tolist()):
            req.output_tokens.append(tok)
            if tok == EOS_TOKEN or len(req.output_tokens) >= req.max_tokens:
                req.finished = True
