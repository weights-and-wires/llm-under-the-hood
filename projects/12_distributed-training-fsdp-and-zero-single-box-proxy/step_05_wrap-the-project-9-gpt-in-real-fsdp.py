"""
Project 12: Step 5 — Wrap the Project 9 GPT in real FSDP

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from nanochat.model import GPT, TransformerBlock  # from Project 9

def setup_fsdp_model(config):
    model = GPT(config)
    wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    sharded = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
    )
    return sharded

model = setup_fsdp_model(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step, batch in enumerate(loader):
    inputs, targets = batch
    logits = model(inputs)
    loss = compute_loss(logits, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
