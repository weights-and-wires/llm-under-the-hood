"""
Project 5: Step 14 — Add checkpoint save/load before you need it

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def save_checkpoint(path: str, model: 'GPT', optimizer, step: int, cfg: 'Config') -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg.__dict__,
    }, path)

def load_checkpoint(path: str, model: 'GPT', optimizer) -> int:
    ckpt = torch.load(path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]
