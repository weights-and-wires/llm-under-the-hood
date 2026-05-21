"""
Project 11: Step 2 — Add per-layer activation histogram

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

activation_stats = {}

def make_hook(name):
    def hook(module, inputs, output):
        x = output.detach().float()
        activation_stats[name] = {
            "mean": x.mean().item(),
            "std": x.std().item(),
            "min": x.min().item(),
            "max": x.max().item(),
            "abs_max": x.abs().max().item(),
        }
    return hook

for name, module in model.named_modules():
    if isinstance(module, TransformerBlock):
        module.register_forward_hook(make_hook(name))
