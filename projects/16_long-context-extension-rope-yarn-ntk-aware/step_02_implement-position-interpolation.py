"""
Project 16: Step 2 — Implement position interpolation

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class RopePI:
    def __init__(self, head_dim, base=10000.0,
                 train_seq_len=1024, target_seq_len=4096):
        self.head_dim = head_dim
        self.base = base
        self.k = target_seq_len / train_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float()
                                   / head_dim))
        self.inv_freq = inv_freq  # frequencies unchanged

    def angles(self, positions):
        # positions: (seq_len,)
        scaled = positions.float() / self.k
        # outer product: (seq_len, head_dim/2)
        return scaled[:, None] * self.inv_freq[None, :]
