"""
Project 12: Step 3 — All-gather and reduce-scatter, drawn out

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def backward_step(self, grad_full_weight: torch.Tensor):
    # grad_full_weight has shape (out_dim, in_dim), same as the full weight.
    grad_flat = grad_full_weight.reshape(-1)
    # Each rank starts with the full gradient (computed locally on its batch).
    # We need reduce-scatter to sum across ranks and keep only this rank's slice.
    out = torch.empty_like(self.flat_shard)
    dist.reduce_scatter_tensor(out, grad_flat, op=dist.ReduceOp.SUM)
    # Now `out` is the summed gradient for this rank's shard.
    self.flat_shard.grad = out
