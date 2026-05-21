"""
Project 12: Step 1 — Launch two PyTorch processes on one box

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )
    x = torch.tensor([float(rank + 1)])
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"rank {rank} saw {x.item()}")
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
