"""Phase-2 lesson 2.6 figure generator (split from the old multi-lesson
make_figures.py during the lesson-centric restructure). Generates the
seasonal markdown price ladder: assets/phase2/lesson2_6_ladder.png."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "assets", "phase2", "lesson2_6_ladder.png")

HORIZON = 12
MAX_STOCK = 15
PRICES = np.array([24, 20, 16, 12])
LAM = np.array([0.4, 0.9, 2.2, 5.0])
GAMMA = 0.95


def main():
    # backward induction (same math as lesson2_6_finite_horizon.py)
    pmf = [poisson.pmf(np.arange(MAX_STOCK + 1), LAM[i]) for i in range(4)]
    for p in pmf:
        p[-1] += 1 - p.sum()
    V = np.zeros((HORIZON + 1, MAX_STOCK + 1))
    pol = np.zeros((HORIZON, MAX_STOCK + 1), dtype=int)
    for t in range(HORIZON - 1, -1, -1):
        pi = pmf[t % 4]
        Q = np.zeros((MAX_STOCK + 1, len(PRICES)))
        for s in range(MAX_STOCK + 1):
            for a, price in enumerate(PRICES):
                ev = 0.0
                for d, pr in enumerate(pi):
                    sold = min(s, d)
                    ev += pr * (price * sold + GAMMA * V[t + 1, s - sold])
                Q[s, a] = ev
        V[t] = Q.max(axis=1)
        pol[t] = Q.argmax(axis=1)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    for s in (0, 5, 10, 15):
        ax.plot(range(HORIZON), pol[:HORIZON, s], "o-",
                label=f"stock={s}")
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"${p:.0f}" for p in PRICES])
    ax.set_xlabel("weeks left → (t=0 is season start; ladder descends)")
    ax.set_ylabel("listed price")
    ax.set_title("Finite-horizon markdown ladder (backward induction)")
    ax.legend(fontsize=8)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
