"""
Project 16: Step 6 — Implement YaRN

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class RopeYaRN:
    def __init__(self, head_dim, base=10000.0,
                 train_seq_len=1024, target_seq_len=4096,
                 alpha=1, beta=32):
        self.head_dim = head_dim
        self.k = target_seq_len / train_seq_len

        # Identify per-pair regime: which pairs have rotated
        # enough times in training to be safe from extension?
        dims = torch.arange(0, head_dim, 2).float()
        freqs = 1.0 / (base ** (dims / head_dim))
        rotations_in_train = freqs * train_seq_len / (2 * math.pi)

        # NTK-by-parts mask: smooth ramp between alpha and beta
        # rotations. Below alpha: full PI. Above beta: no change.
        mask = torch.clamp((rotations_in_train - alpha)
                           / (beta - alpha), 0.0, 1.0)
        self.inv_freq = freqs * (mask + (1 - mask) / self.k)

        # YaRN attention temperature
        self.attn_temperature = 0.1 * math.log(self.k) + 1.0

    def angles(self, positions):
        return positions.float()[:, None] * self.inv_freq[None, :]

    def scale_attention(self, attn_logits):
        return attn_logits / self.attn_temperature

model.set_rope(RopeYaRN(head_dim=64, base=10000.0,
                       train_seq_len=1024, target_seq_len=4096))
yarn_ppl = {}
for L in context_lengths:
    yarn_ppl[L] = evaluate_perplexity(model, eval_dataset, seq_len=L)
    print(f"YaRN: length={L}, ppl={yarn_ppl[L]:.2f}")
