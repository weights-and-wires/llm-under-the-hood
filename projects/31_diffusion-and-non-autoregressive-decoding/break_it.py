"""
Project 31: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

print("baseline, one blank per step:")
diffusion_decode("the weather today is", n_blanks=6, steps=6)

print("\nbroken, all blanks in one step:")
diffusion_decode("the weather today is", n_blanks=6, steps=1)

for T in (1, 2, 3, 6):
    out = diffusion_decode("the weather today is", n_blanks=6,
                           steps=T, verbose=False)
    print(f"steps={T}:  {out}")
