"""Lesson 4.6 figure: equal mean, different tail — why CVaR changes the answer.

Left: two return distributions with (nearly) the same mean but very
different left tails — Gaussian-like vs jump-heavy (the lesson's
demand-jump environment). Right: the CVaR5% grid over order-up-to level S*
from the lesson's own experiment (mean-optimal S*=6 vs CVaR-optimal S*=4).
Re-runs phase4_hybrid/lesson4_6_risk_sensitive.py's grid logic (seeded).
Run:  python phase2_mdp/make_figures_46.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "assets", "phase4")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "phase4_hybrid"))

ALPHA = 0.05


def cvar(samples, alpha=ALPHA):
    """CVaR_alpha = mean of the worst (1-alpha) tail."""
    k = max(1, int((1 - alpha) * len(samples)))
    return np.sort(samples)[:k].mean()


def cvar_grid():
    """Re-run the lesson's CVaR grid scan (import its machinery).
    NOTE: the lesson MAXIMIZES CVaR of returns (CVaR of the worst 5% of
    episodes); replicate exactly: tail = worst 5%, value = tail.mean(),
    higher = better."""
    import lesson4_6_risk_sensitive as L
    rng = np.random.default_rng(99)
    alpha = L.ALPHA
    results = {}
    for S_star in range(L.MAX_INV + 1):
        Gs = np.array([L.rollout(L.order_up_to(S_star), rng,
                                 horizon=L.EP_LEN)
                       for _ in range(4000)])
        tail = np.sort(Gs)[: max(1, int(alpha * len(Gs)))]
        results[S_star] = (Gs.mean(), tail.mean())
    return results


def main():
    # ---- left panel: same mean, different tail
    rng = np.random.default_rng(5)
    n = 200_000
    base = rng.normal(100, 25, n)
    jumps = rng.random(n) < 0.02
    risky = np.where(jumps, rng.normal(100, 25, n) - rng.uniform(60, 90, n),
                     rng.normal(102.7, 20, n))
    # equalize means exactly for honesty
    risky = risky - risky.mean() + base.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    ax = axes[0]
    bins = np.linspace(0, 200, 90)
    ax.hist(base, bins=bins, density=True, alpha=0.6, color="#31a354",
            label=f"stable market (CVaR5% = {cvar(base):.1f})")
    ax.hist(risky, bins=bins, density=True, alpha=0.55, color="#d94801",
            label=f"jump-risk market (CVaR5% = {cvar(risky):.1f})")
    ax.axvline(base.mean(), color="black", ls="--", lw=1.2,
               label=f"equal mean = {base.mean():.1f}")
    ax.axvline(cvar(risky), color="#d94801", ls=":", lw=1.4)
    ax.axvline(cvar(base), color="#31a354", ls=":", lw=1.4)
    ax.set_xlabel("episode return")
    ax.set_ylabel("density")
    ax.set_title("same mean, different left tail — mean is blind, CVaR is not")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---- right panel: CVaR grid from the lesson's environment
    res = cvar_grid()
    ss = sorted(res)
    means = [res[s][0] for s in ss]
    cvars_ = [res[s][1] for s in ss]
    ax = axes[1]
    ax.plot(ss, means, "o-", color="#31a354", lw=1.5,
            label="mean return (mean-opt S*=6)")
    ax.plot(ss, cvars_, "s-", color="#d94801", lw=1.5,
            label="CVaR 5% (CVaR-opt S*=4)")
    best_mean = int(np.argmax(means))
    best_cvar = int(np.argmax(cvars_))
    ax.annotate(f"mean-opt S*={ss[best_mean]}", (ss[best_mean],
                means[best_mean]), textcoords="offset points",
                xytext=(8, 10), fontsize=8, color="#31a354")
    ax.annotate(f"CVaR-opt S*={ss[best_cvar]}", (ss[best_cvar],
                cvars_[best_cvar]), textcoords="offset points",
                xytext=(8, -14), fontsize=8, color="#d94801")
    ax.set_xlabel("order-up-to level S*")
    ax.set_ylabel("value")
    ax.set_title("the risk measure chooses the policy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, "lesson4_6_cvar.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))
    print("mean-opt S*:", ss[best_mean], "| CVaR-opt S*:", ss[best_cvar])


if __name__ == "__main__":
    main()
