"""
Project 5: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

with open(log_path, "a", encoding="utf-8") as f:
    f.write(f"{step},{train_loss:.4f},{val_loss:.4f},{lr:.8f},{grad_norm:.4f}\n")

self.lm_head.weight = self.token_embedding.weight

for param_group in optimizer.param_groups:
    param_group["lr"] = cfg.learning_rate

torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.zeros_(module.weight)
