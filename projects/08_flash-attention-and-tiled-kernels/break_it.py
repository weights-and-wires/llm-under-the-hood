"""
Project 8: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

m_tile = s.max(dim=-1, keepdim=True).values
p = torch.exp(s - m_tile)
l_tile = p.sum(dim=-1, keepdim=True)

# m_tile = s.max(dim=-1, keepdim=True).values   # removed
p = torch.exp(s)                                # no stability shift
l_tile = p.sum(dim=-1, keepdim=True)

# m_new = torch.maximum(m, m_tile)
# alpha = torch.exp(m - m_new)
# beta = torch.exp(m_tile - m_new)
# l = alpha * l + beta * l_tile
# o = alpha * o + beta * (p @ v_tile)
# m = m_new
l = l + l_tile
o = o + (p @ v_tile)
