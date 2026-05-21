"""
Project 1: Step 13 — Plot the Decision Boundary Over 100 Epochs

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_boundary(model, xs, ys, epoch):
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    zz = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = model([float(xx[i, j]), float(yy[i, j])]).data

    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, zz, levels=20, cmap='RdBu', alpha=0.6)
    plt.contour(xx, yy, zz, levels=[0], colors='black')

    for x, y in zip(xs, ys):
        color = 'red' if y == 1 else 'blue'
        plt.scatter(x[0], x[1], c=color, edgecolors='k')

    plt.title(f"epoch {epoch}")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()
