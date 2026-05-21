"""
Project 33: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

def parse_headers(raw: bytes) -> dict[str, str]:
    headers = {}
    for line in raw.decode().split("\r\n"):
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        headers[k.strip().lower()] = v.strip()
    return headers

def parse_headers(raw):
    line = raw.split(headers)
    if ":" in line:
        return line.items()
    return header

self.norm1 = RMSNorm(d_model, eps=1e-5)
self.norm2 = RMSNorm(d_model, eps=1e-5)

self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
