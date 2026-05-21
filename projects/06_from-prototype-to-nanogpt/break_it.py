"""
Project 6: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

x = x + self.attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))

x = self.attn(self.ln1(x))
x = self.mlp(self.ln2(x))

x = x + self.attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))

x = x + self.attn(x)
x = x + self.mlp(x)
