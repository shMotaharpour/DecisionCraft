"""Lesson 1.2b figure: symmetry — why two identical solutions cost double.

Toy model: choose 2 of B identical warehouses (same costs). Without
symmetry breaking the solver sees C(B,2) equally-optimal solutions (the
orbit); with an ordering constraint (z_i >= z_{i+1}) the orbit collapses
to 1.
Left: the 6 optimal selections for B=4 as a grid (the symmetry orbit).
Right: orbit size vs B, log scale — C(B,2) vs 1.
Run:  python phase1_milp/make_figures_12b.py
"""
import itertools
import os
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "phase1")
os.makedirs(OUT, exist_ok=True)

N = 4
K = 2


def main():
    orbit = list(itertools.combinations(range(N), K))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    ax = axes[0]
    rows = (len(orbit) + 1) // 2
    for idx, combo in enumerate(orbit):
        # fill row-major top-down so solutions read 1..6 in order
        col = idx % 2
        row = idx // 2
        y = (rows - 1 - row) * 1.6
        x = col * 2.4
        for i in range(N):
            on = i in combo
            ax.add_patch(plt.Rectangle((x + i * 0.5, y), 0.45, 0.8,
                                       facecolor="#31a354" if on
                                       else "#eeeeee",
                                       edgecolor="#555555", lw=1.0))
        ax.text(x + 1.0, y + 1.25, f"solution {idx+1}", ha="center",
                fontsize=8)
    ax.set_xlim(-0.2, 4.6)
    ax.set_ylim(-0.5, rows * 1.6 + 0.5)
    ax.axis("off")
    ax.set_title(f"{N} identical warehouses, pick {K}: "
                 f"{len(orbit)} equally-optimal solutions", pad=12)

    ax = axes[1]
    ns = list(range(4, 15))
    ax.plot(ns, [comb(n, K) for n in ns], "o-", color="#d62728", lw=1.5,
            label="search space without ordering — C(B, 2)")
    ax.plot(ns, [1] * len(ns), "s-", color="#31a354", lw=1.5,
            label="with z₁ ≥ z₂ ≥ … (orbit collapsed to 1)")
    ax.set_yscale("log")
    ax.set_xlabel("identical warehouses B")
    ax.set_ylabel("distinct optimal solutions the solver may visit (log)")
    ax.set_title("symmetry breaking collapses the orbit to one")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(OUT, "lesson1_2b_symmetry.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))
    print("orbit size for B=4:", len(orbit))


if __name__ == "__main__":
    main()
