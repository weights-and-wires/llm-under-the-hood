"""
Project 17: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

while len(self.running) < self.max_batch_size and self.waiting:
    new_req = self.waiting.popleft()
    self._prefill(new_req)
    self.running.append(new_req)
