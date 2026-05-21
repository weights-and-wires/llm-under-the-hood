"""
Project 12: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

shard_size = total // world_size
self.shard_start = rank * shard_size
self.shard_end = (rank + 1) * shard_size

shard_size = total // world_size
self.shard_start = rank * shard_size
if rank == 1:
    self.shard_start += 1  # off-by-one: this rank's shard is shifted by 1
self.shard_end = self.shard_start + shard_size
