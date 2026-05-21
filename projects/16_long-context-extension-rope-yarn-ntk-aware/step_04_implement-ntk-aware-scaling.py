"""
Project 16: Step 4 — Implement NTK-aware scaling

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class RopeNTK:
    def __init__(self, head_dim, base=10000.0,
                 train_seq_len=1024, target_seq_len=4096):
        self.head_dim = head_dim
        k = target_seq_len / train_seq_len
        # NTK-aware base adjustment: stretches the slow pairs,
        # leaves the fast pairs nearly untouched.
        new_base = base * (k ** (head_dim / (head_dim - 2)))
        inv_freq = 1.0 / (new_base ** (torch.arange(0, head_dim, 2).float()
                                       / head_dim))
        self.inv_freq = inv_freq  # frequencies modified

    def angles(self, positions):
        return positions.float()[:, None] * self.inv_freq[None, :]
