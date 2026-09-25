"""Lesson 1.9b figure: the Lagrangian dual landscape + subgradient walk.

Left  — the dual function L(mu) sampled over mu, with the block-choice
        breakpoints marked; the minimized optimum (best bound) vs the
        MILP optimum line shows the duality gap as a vertical segment.
Right — the subgradient walk itself: mu trajectory and L(mu) per
        iteration converging to the dual optimum; overuse iterations
        (g<0) push mu up, slack iterations pull it down.

Run:  python phase1_milp/make_figures_19b.py   (~15 s, seeded)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(os.path.dirname(__file__), "assets", "lesson1_9b_lagrangian.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

NB, NJ, CAP, LOCAL = 2, 10, 14, 4
rng = np.random.default_rng(19)
PROFIT = rng.integers(5, 40, size=(NB, NJ)).astype(float)
USE = rng.integers(1, 6, size=(NB, NJ)).astype(float)


def blocks_solve(mu):
    picks, lag = [], mu * CAP
    for b in range(NB):
        red = PROFIT[b] - mu * USE[b]
        chosen = [int(j) for j in np.argsort(-red)
                  if red[j] > 0][:LOCAL]
        picks.append(chosen)
        lag += float(red[chosen].sum())
    return lag, picks


def total_use(picks):
    return sum(USE[b, j] for b in range(NB) for j in picks[b])


def best_feasible(picks):
    flat = sorted(((PROFIT[b, j] / USE[b, j], b, j)
                   for b in range(NB) for j in picks[b]), reverse=True)
    tot_use = tot_p = 0.0
    for _, b, j in flat:
        if tot_use + USE[b, j] <= CAP:
            tot_use += USE[b, j]; tot_p += PROFIT[b, j]
    return tot_p


def main():
    # ground truth MILP (same as the lesson script)
    from scipy.optimize import milp, LinearConstraint, Bounds
    nv = NB * NJ
    A = np.vstack([USE.flatten().reshape(1, -1),
                   np.vstack([np.concatenate([np.ones(NJ), np.zeros(NJ)]),
                              np.concatenate([np.zeros(NJ), np.ones(NJ)])])])
    ub = np.array([CAP, LOCAL, LOCAL])
    res = milp(c=-PROFIT.flatten(),
               constraints=[LinearConstraint(A, -np.inf, ub)],
               integrality=np.ones(nv), bounds=Bounds(0, 1))
    milp_opt = float(-res.fun)
    milp_use = float(USE.flatten() @ res.x)

    # dual landscape: L(mu) over a mu grid
    mus = np.linspace(0, 12, 241)
    L_vals = np.array([blocks_solve(float(m))[0] for m in mus])
    dual_opt = float(L_vals.min())
    mu_star = float(mus[L_vals.argmin()])
    gap_pct = (dual_opt - milp_opt) / milp_opt * 100

    # subgradient walk (same dynamics as the lesson script)
    mu, step0 = 5.0, 1.5
    walk_mu, walk_L, walk_g = [], [], []
    for it in range(300):
        lag, picks = blocks_solve(mu)
        g = CAP - total_use(picks)
        walk_mu.append(mu); walk_L.append(lag); walk_g.append(g)
        mu = max(0.0, mu - step0 * g / (1 + it * 0.03))
        if abs(g) < 1e-3 and it > 20:
            break
    walk_best = min(walk_L)
    print(f"milp={milp_opt:.2f} (use {milp_use:.1f})  dual_opt={dual_opt:.2f} "
          f"at mu*={mu_star:.2f}  gap={gap_pct:.2f}%")
    print(f"walk: {len(walk_L)} iters, best L={walk_best:.2f}, "
          f"final mu={walk_mu[-1]:.3f}")
    assert abs(walk_best - dual_opt) / dual_opt < 0.02, \
        "subgradient walk must land within 2% of the grid-search optimum"
    # NOTE: this figure uses its own fresh rng(19) instance (the lesson
    # script consumes rng in parts a/b before c, so its part-c instance
    # differs: MILP 140, bound 141.25, gap 0.89%). Same generator, same
    # dynamics — a different draw. Both measured; evidence documents both.

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    ax = axes[0]
    ax.plot(mus, L_vals, color="#08519c", lw=1.8, label="dual function L(μ)")
    # breakpoints where block argmin sets change (kinks of the PW-linear L)
    dL = np.diff(L_vals)
    dd = np.abs(np.diff(dL))
    kinks = mus[1:-1][dd > 1e-6][:12]
    for k in kinks:
        ax.axvline(k, color="gray", lw=0.4, alpha=0.5)
    ax.axhline(milp_opt, color="#31a354", lw=1.4, ls="--",
               label=f"MILP optimum {milp_opt:.1f}")
    ax.plot([mu_star], [dual_opt], "o", color="#e6550d", ms=7, zorder=5)
    ax.text(0.98, 0.55,
            f"dual optimum {dual_opt:.1f} at μ*={mu_star:.2f}"
            f"\nduality gap {gap_pct:.2f}% above MILP",
            transform=ax.transAxes, fontsize=8, color="#e6550d",
            ha="right", va="top")
    ax.set_xlabel("Lagrange multiplier μ (machine price)")
    ax.set_ylabel("L(μ)")
    ax.set_title("The Lagrangian dual: convex, piecewise-linear, minimized")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    its = np.arange(len(walk_L))
    ax.plot(its, walk_L, color="#08519c", lw=1.5, label="L(μ_t) per iteration")
    ax.axhline(dual_opt, color="#e6550d", lw=1.2, ls="--",
               label=f"dual optimum {dual_opt:.1f}")
    ax.axhline(milp_opt, color="#31a354", lw=1.2, ls=":",
               label=f"MILP {milp_opt:.1f}")
    ax2 = ax.twinx()
    ax2.plot(its, walk_mu, color="#756bb1", lw=1.2, alpha=0.85,
             label="μ_t trajectory")
    ax2.set_ylabel("μ (resource price)", color="#756bb1")
    ax.set_xlabel("subgradient iteration")
    ax.set_ylabel("L(μ_t)")
    ax.set_title("Subgradient descent finds the dual optimum")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")
    print("data:", {"dual_opt": round(dual_opt, 2), "mu_star": round(mu_star, 2),
                    "gap_pct": round(gap_pct, 2),
                    "walk_iters": len(walk_L),
                    "walk_best": round(walk_best, 2),
                    "kinks": len(kinks)})


if __name__ == "__main__":
    main()
