"""
Project 12: Step 2 — Implement parameter sharding by hand

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class HandShardedLinear:
    def __init__(self, in_dim: int, out_dim: int, rank: int, world_size: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.world_size = world_size
        total = in_dim * out_dim
        assert total % world_size == 0, "shard must divide evenly"
        shard_size = total // world_size
        self.shard_start = rank * shard_size
        self.shard_end = (rank + 1) * shard_size
        # Each rank initializes only its own slice.
        torch.manual_seed(0)  # deterministic across ranks
        full_init = torch.randn(total) * 0.02
        self.flat_shard = full_init[self.shard_start:self.shard_end].clone()
        self.flat_shard.requires_grad_(True)

def gather_full_weight(self) -> torch.Tensor:
    pieces = [torch.empty_like(self.flat_shard) for _ in range(self.world_size)]
    dist.all_gather(pieces, self.flat_shard)
    full_flat = torch.cat(pieces, dim=0)
    return full_flat.view(self.out_dim, self.in_dim)

def forward(self, x: torch.Tensor) -> torch.Tensor:
    W = self.gather_full_weight()
    y = x @ W.t()
    # Important: in real FSDP the full W is freed immediately after use,
    # so it does not stay in memory beyond this scope.
    return y
