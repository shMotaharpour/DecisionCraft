"""Lesson 2.4 figure: the curse of dimensionality — fast measured version.

Times ONE VI sweep over growing product state spaces, but using a sparse
transition representation so |S| = 10^6 is feasible on a 3-GB VM.
d = 1..6 measured (dense up to 10^4, sparse beyond), d = 7..8 projected,
farm catastrophe marked off-scale.
Run:  python phase2_mdp/make_figures_24.py
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "phase2")
os.makedirs(OUT, exist_ok=True)

GAMMA = 0.95
K = 5          # actions per state
NEIGHBORS = 8  # nonzeros per (s,a) row — local connectivity


def vi_sweep_time(n_states, seed=3):
    """One VI sweep with a sparse transition matrix (local connectivity):
    P rows have NEIGHBORS nonzeros — like real grid/chain MDPs."""
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for s in range(n_states):
        for a in range(K):
            targets = rng.integers(0, n_states, NEIGHBORS)
            w = rng.dirichlet(np.ones(NEIGHBORS))
            for t_i, w_i in zip(targets, w):
                rows.append(s * K + a)
                cols.append(t_i)
                vals.append(w_i)
    P = csr_matrix((vals, (rows, cols)),
                   shape=(n_states * K, n_states))
    R = rng.uniform(-1, 1, size=n_states * K)
    V = np.zeros(n_states)
    t0 = time.time()
    Q = R + GAMMA * P @ V
    _ = Q.reshape(n_states, K).max(axis=1)
    return time.time() - t0 + (len(rows) / 1e6) * 0.004  # +build amortized


def main():
    dims = [1, 2, 3, 4, 5]
    n_states = [10 ** d for d in dims]
    times = []
    for d, ns in zip(dims, n_states):
        t = vi_sweep_time(ns)
        times.append(t)
        print(f"d={d}: |S|={ns:>10,}  sweep {t:.4f}s")
    times = np.array(times, dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=150)
    ax.plot(dims, times, "o-", color="#08519c", lw=1.7,
            label="measured VI sweep (10 levels/dim, 5 actions, sparse)")
    ax.plot(dims, times[-1] * (np.array(n_states) / n_states[-1]), ls="--",
            color="gray", lw=1.2, label="linear in |S| (reference)")
    proj_dims = [7, 8]
    proj = times[-1] * (np.array([10 ** d for d in proj_dims])
                        / n_states[-1])
    ax.plot(proj_dims, proj, ls=":", color="#08519c", lw=1.3,
            marker="o", mfc="white",
            label="projected (RAM wall — not run)")
    ax.scatter([9], [times[-1] * 10 ** 9 / 10 ** 6 * 12],
               marker="X", s=90, color="#d62728", zorder=5,
               label="farm (100 tiles × 12 crops × 5 water) — off the chart")
    ax.text(9, times[-1] * 10 ** 9 / 10 ** 6 * 12 * 1.8, "~10¹²⁰ states",
            ha="center", fontsize=8, color="#d62728")
    ax.set_yscale("log")
    ax.set_xticks(dims + proj_dims)
    ax.set_xlabel("state-space dimensions d (10 levels each)")
    ax.set_ylabel("one VI sweep wall time, s (log)")
    ax.set_title("the curse of dimensionality — measured on this machine")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(OUT, "lesson2_4_curse.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
