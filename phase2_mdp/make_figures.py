"""Phase-2 figures for the DecisionCraft course notes.

Every figure is generated from the same seeded logic as its lesson script,
so figures and evidence files stay consistent. Outputs land in
assets/phase2/. Placement rule (set by review): DATA figures go AFTER the
numbers they visualize; conceptual diagrams go before the discussion.
Run:  python phase2_mdp/make_figures.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "phase2")
OUT_P1 = os.path.join(HERE, "..", "assets", "phase1")
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT_P1, exist_ok=True)

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


# ------------------------------------------------------- lesson 1.6 figure
def fig_crossover():
    """Lesson 1.6 fig: exact MILP vs heuristic wall time + gap across n."""
    import time
    from scipy.optimize import milp, LinearConstraint, Bounds

    def make_instance(n, seed=7):
        rng = np.random.default_rng(seed)
        w = rng.integers(2, 41, n)
        p = rng.integers(5, 91, n)
        return list(w), list(p), int(w.sum() // 3)

    def solve_exact(w, p, cap):
        n = len(w)
        c = -np.array(p, dtype=float)
        A = np.array(w, dtype=float).reshape(1, -1)
        res = milp(c=c, constraints=LinearConstraint(A, -np.inf, cap),
                   integrality=np.ones(n), bounds=Bounds(0, 1),
                   options={"mip_rel_gap": 0.0})
        return -res.fun if res.success else None

    def local_search(w, p, cap, budget_s=0.5, seed=1):
        rng = np.random.default_rng(seed)
        bits = [0] * len(w)
        tw = 0
        order = sorted(range(len(w)), key=lambda i: -p[i] / w[i])
        for i in order:
            if tw + w[i] <= cap:
                bits[i], tw = 1, tw + w[i]
        val = sum(x for x, b in zip(p, bits) if b)
        t0 = time.time()
        while time.time() - t0 < budget_s:
            i = int(rng.integers(len(bits)))
            cand, cw = bits[:], tw
            cand[i] = 1 - cand[i]
            cw += w[i] * (1 if cand[i] else -1)
            if cw <= cap:
                cv = val + p[i] * (1 if cand[i] else -1)
                if cv > val:
                    bits, val = cand, cv
        return val

    ns = [50, 100, 200, 400, 800, 1600, 3200, 6400]
    t_exact, t_heur, gaps = [], [], []
    for n in ns:
        w, p, cap = make_instance(n)
        t0 = time.time()
        opt = solve_exact(w, p, cap)
        t_exact.append(time.time() - t0)
        t0 = time.time()
        hv = local_search(w, p, cap)
        t_heur.append(time.time() - t0)
        gaps.append(100 * (opt - hv) / opt)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    ax = axes[0]
    ax.plot(ns, t_exact, "o-", color="#08519c", lw=1.6, label="scipy.milp exact (gap=0)")
    ax.plot(ns, t_heur, "s-", color="#d94801", lw=1.6,
            label="greedy + local search (0.5 s fixed)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("items n (log)")
    ax.set_ylabel("wall time s (log)")
    ax.set_title("knapsack: exact vs heuristic wall time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.bar([str(n) for n in ns], gaps, color="#31a354", alpha=0.85)
    for i, g in enumerate(gaps):
        ax.text(i, g + 0.008, f"{g:.2f}", ha="center", fontsize=7)
    ax.set_xlabel("items n")
    ax.set_ylabel("optimality gap %")
    ax.set_title("heuristic gap vs exact optimum (lower = better)")
    ax.set_ylim(0, max(gaps) * 1.3)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(OUT_P1, "lesson1_6_crossover.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))


if __name__ == "__main__":
    fig_markdown_ladder()
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        fig_crossover()
