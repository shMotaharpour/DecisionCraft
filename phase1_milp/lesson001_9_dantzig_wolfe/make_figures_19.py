"""Lesson 1.9 figure: Dantzig-Wolfe vs direct LP as the block count grows.

Same instance family as phase1_milp/lesson1_9_dantzig_wolfe.py (seeded):
direct block-diagonal LP vs DW + column generation, for B = 4..60 blocks.
Left: wall time (the decomposition's payoff). Right: optimality match —
DW must land exactly on the direct optimum every time (it's the same LP).
Run:  python phase2_mdp/make_figures_19.py
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "assets", "phase1")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "phase1_milp"))

from lesson1_9_dantzig_wolfe import dw_solve, make_instance, solve_direct


def main():
    block_counts = [4, 8, 12, 20, 30, 45, 60]
    t_direct, t_dw, gaps = [], [], []
    for B in block_counts:
        blocks, gu, gc = make_instance(B=B)
        t0 = time.time()
        opt_direct, _ = solve_direct(blocks, gu, gc)
        t_direct.append(time.time() - t0)
        opt_dw, iters, ncols, t_dw_s, _ = dw_solve(blocks, gu, gc,
                                                   verbose=False)
        t_dw.append(t_dw_s)
        gaps.append(abs(opt_direct - opt_dw))
        print(f"B={B:3d}: direct {t_direct[-1]:6.3f}s | DW {t_dw[-1]:6.3f}s "
              f"({iters} rounds, {ncols} cols) | |diff| {gaps[-1]:.1e}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    x = np.log2(block_counts)
    ax = axes[0]
    ax.plot(x, t_direct, "o-", color="#08519c", lw=1.6,
            label="direct LP (one big system)")
    ax.plot(x, t_dw, "s--", color="#d94801", lw=1.6,
            label="Dantzig-Wolfe + column generation")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in block_counts])
    ax.set_xlabel("blocks B (8 activities each, log2 spacing)")
    ax.set_ylabel("wall time s")
    ax.set_yscale("log")
    ax.set_title("direct LP vs DW+CG wall time (toy scale: direct wins)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(x, np.maximum(gaps, 1e-15), "o", color="#31a354", ms=6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in block_counts])
    ax.set_xlabel("blocks B")
    ax.set_ylabel("|direct − DW| (log)")
    ax.set_title("optimality certificate at every B")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(OUT, "lesson1_9_dw_scale.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
