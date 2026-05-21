"""
Project 18: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

dead = torch.tensor([0, 0, 1, 1], device=x.device, dtype=torch.bool)
router_logits = self.router(x)
router_logits[..., dead] = -1e9

for p in self.experts[2].parameters():
    p.requires_grad = False
for p in self.experts[3].parameters():
    p.requires_grad = False

dead = torch.tensor([0, 0, 0, 1], device=x.device, dtype=torch.bool)
router_logits[..., dead] = -1e9
