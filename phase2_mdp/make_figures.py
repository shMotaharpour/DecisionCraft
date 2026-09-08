"""Phase-2 figures for the DecisionCraft course notes.

Every figure is generated from the same seeded logic as its lesson script,
so figures and evidence files stay consistent. Outputs land in
assets/phase2/. Placement rule (set by review): DATA figures go AFTER the
numbers they visualize; conceptual diagrams go before the discussion.
Run:  python phase2_mdp/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "phase2")
os.makedirs(OUT, exist_ok=True)

HORIZON = 12
MAX_STOCK = 15
PRICES = np.array([24, 20, 16, 12])
LAM = np.array([0.4, 0.9, 2.2, 5.0])
GAMMA = 0.95


def demand_pmf(k):
    pmf = poisson.pmf(np.arange(MAX_STOCK + 1), LAM[k])
    pmf[-1] += 1.0 - pmf.sum()
    return pmf


PMFS = [demand_pmf(k) for k in range(len(PRICES))]


def backward_induction():
    V = np.zeros((HORIZON + 1, MAX_STOCK + 1))
    policy = np.zeros((HORIZON, MAX_STOCK + 1), dtype=int)
    for t in range(HORIZON - 1, -1, -1):
        for s in range(MAX_STOCK + 1):
            best_v, best_a = -1e9, 0
            for k, pmf in enumerate(PMFS):
                ev = sum(p * (PRICES[k] * min(s, d) + V[t + 1][max(0, s - d)])
                         for d, p in enumerate(pmf))
                if ev > best_v:
                    best_v, best_a = ev, k
            V[t][s] = best_v
            policy[t][s] = best_a
    return V, policy


def stationary_values():
    vals = []
    for k in range(len(PRICES)):
        Vs = np.zeros((HORIZON + 1, MAX_STOCK + 1))
        for t in range(HORIZON - 1, -1, -1):
            for s in range(MAX_STOCK + 1):
                Vs[t][s] = sum(p * (PRICES[k] * min(s, d) + Vs[t + 1][max(0, s - d)])
                               for d, p in enumerate(PMFS[k]))
        vals.append(Vs[0][MAX_STOCK])
    return np.array(vals)


def fig_markdown_ladder():
    """Lesson 2.6 fig 1: the optimal price ladder as a heatmap + value path."""
    V, policy = backward_induction()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)

    ax = axes[0]
    cmap = matplotlib.colors.ListedColormap(
        ["#f7f7f7", "#fee0d2", "#fc9272", "#de2d26", "#a50f15"])
    im = ax.imshow(policy, cmap=cmap, vmin=-0.5, vmax=4.5, aspect="auto",
                   origin="upper")
    ax.set_xticks(range(MAX_STOCK + 1))
    ax.set_xticklabels(range(MAX_STOCK + 1), fontsize=7)
    ax.set_yticks(range(HORIZON))
    ax.set_yticklabels([f"t={t}" for t in range(HORIZON)], fontsize=7)
    ax.set_xlabel("units in stock s")
    ax.set_ylabel("weeks remaining t")
    ax.set_title("optimal price ladder (backward induction)")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.04)
    cbar.ax.set_yticklabels([f"P{p}" for p in PRICES], fontsize=8)

    ax = axes[1]
    ax.plot(range(HORIZON + 1), V[:, MAX_STOCK], "o-", color="#08519c",
            lw=1.6, ms=4, label="V_t (15 in stock)")
    ax.plot(range(HORIZON + 1), V[:, 8], "s--", color="#6baed6",
            lw=1.2, ms=3, label="V_t (8 in stock)")
    stat = stationary_values()
    ax.axhline(stat.max(), color="#d94801", ls=":", lw=1.4,
               label=f"best stationary (P{PRICES[stat.argmax()]}) = "
                     f"{stat.max():.1f}")
    ax.annotate(f"V_0 = {V[0][MAX_STOCK]:.1f}\n(+9.2% vs stationary)",
                xy=(0, V[0][MAX_STOCK]), xytext=(2.2, V[0][MAX_STOCK] + 12),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.set_xlabel("weeks remaining t")
    ax.set_ylabel("value")
    ax.set_title("finite-horizon value vs the stationary ceiling")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, "lesson2_6_ladder.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))


if __name__ == "__main__":
    fig_markdown_ladder()
