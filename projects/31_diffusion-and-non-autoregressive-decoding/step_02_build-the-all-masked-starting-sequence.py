"""
Project 31: Step 2 — Build the all-masked starting sequence

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def build_sequence(prefix_text, n_blanks):
    prefix = tok(prefix_text, add_special_tokens=False)["input_ids"]
    ids = ([tok.cls_token_id] + prefix
           + [MASK] * n_blanks + [tok.sep_token_id])
    x = torch.tensor([ids])                       # shape (1, seq_len)
    start = 1 + len(prefix)                        # first blank position
    fill = torch.arange(start, start + n_blanks)   # positions we may fill
    return x, fill

x, fill = build_sequence("the weather today is", n_blanks=6)
print(tok.decode(x[0]))
# [CLS] the weather today is [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [SEP]
