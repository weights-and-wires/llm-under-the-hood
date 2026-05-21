"""
Project 15: Step 6 — Apply attention the same way as before

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
y = att @ v

y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.d_head)
y = self.o_proj(y)
