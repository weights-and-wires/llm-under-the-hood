"""
Project 19: Step 6 — Fit the scaling law

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import numpy as np
import pandas as pd

df = pd.read_csv("scaling_runs.csv")
x = np.log(df["params"].to_numpy())
y = np.log(df["val_loss"].to_numpy())

slope, intercept = np.polyfit(x, y, 1)
b = -slope
a = np.exp(intercept)

print("a =", a)
print("b =", b)
