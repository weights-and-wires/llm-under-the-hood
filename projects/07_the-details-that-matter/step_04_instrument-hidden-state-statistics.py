"""
Project 7: Step 4 — Instrument hidden-state statistics

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def tensor_stats(x: torch.Tensor) -> dict:
    flat = x.detach().float().reshape(-1)
    mean = flat.mean()
    var = flat.var(unbiased=False)
    std = torch.sqrt(var + 1e-12)
    centered = flat - mean
    kurt = (centered ** 4).mean() / (std ** 4 + 1e-12)

    return {
        "mean": mean.item(),
        "var": var.item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "kurtosis": kurt.item(),
    }

if step % config.stats_every == 0:
    stats["block_03_pre_attn"] = tensor_stats(x)
