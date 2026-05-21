"""
Project 19: Step 5 — Plot loss versus parameter count on log axes

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("scaling_runs.csv")  # columns: depth, params, val_loss
plt.scatter(df["params"], df["val_loss"])
plt.xscale("log")
plt.xlabel("Parameter count")
plt.ylabel("Final validation loss")
plt.show()
