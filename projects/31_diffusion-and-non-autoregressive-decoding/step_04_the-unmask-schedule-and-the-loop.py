"""
Project 31: Step 4 — The unmask schedule and the loop

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

@torch.no_grad()
def diffusion_decode(prefix_text, n_blanks, steps, verbose=True):
    x, fill = build_sequence(prefix_text, n_blanks)
    committed = torch.zeros(n_blanks, dtype=torch.bool)

    for s in range(steps):
        conf, pred = predict(x, fill)
        conf = conf.masked_fill(committed, -1.0)   # ignore filled blanks

        target = round(n_blanks * (s + 1) / steps)  # how many filled by now
        reveal = target - int(committed.sum())
        if reveal <= 0:
            continue

        order = torch.argsort(conf, descending=True)  # most confident first
        for j in order[:reveal]:
            x[0, fill[j]] = pred[j]     # write the token into the blank
            committed[j] = True         # mark it committed

        if verbose:
            filled = int(committed.sum())
            print(f"step {s+1:>2}/{steps}  filled {filled}/{n_blanks}  "
                  f"|  {tok.decode(x[0, fill])}")

    return tok.decode(x[0, fill])

out = diffusion_decode("the weather today is", n_blanks=6, steps=6)
