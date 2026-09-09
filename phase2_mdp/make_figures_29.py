"""Lesson 2.9 figure: reward design + terminal value, side by side.

Left — terminal value: last-week order-up-to levels for the three
conventions (zero/liquidate/carryover) against days-to-go; the end
bends the policy backward. Right — reward design: TRUE performance of
Q-learning policies trained on each signal vs the exact optimum line.

Run: python phase2_mdp/make_figures_29.py   (~4 min, seeded)
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesson2_9a_reward_design import (GAMMA, MAX_S, PMF, PRICE, COST, HOLD,
                                     LAM, evaluate, run_qlearning,
                                     true_reward)
from lesson2_9b_terminal_value import SALVAGE_TRUE, solve, true_eval

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "phase2",
                   "lesson2_9_reward_terminal.png")


def main():
    # ---- left: terminal-value policies ----
    DAYS = 30
    TV = {
        "zero (worthless)": (lambda s: 0.0, "#e6550d"),
        "liquidate (salvage)": (lambda s: SALVAGE_TRUE * s, "#31a354"),
        "carryover (full margin)": (lambda s: (PRICE - COST) * s, "#756bb1"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    ax = axes[0]
    for name, (tv, color) in TV.items():
        V, pol = solve(tv)
        last_week = [int(pol[t][0]) for t in range(DAYS - 7, DAYS)]
        ax.plot(range(DAYS - 7, DAYS + 1), last_week + [last_week[-1]],
                "o-", color=color, lw=1.8, ms=5, label=name)
    ax.set_xlabel("day (last week of the season)")
    ax.set_ylabel("order-up-to level at s=0")
    ax.set_title("Terminal value bends the policy backward")
    ax.legend(fontsize=8, loc="center left")
    ax.set_xticks(range(DAYS - 7, DAYS + 1))

    # ---- right: reward design (retrain all three, same seeds) ----
    t0 = time.time()
    r_sparse = lambda s, a, d, s2: true_reward(s, a, d)
    r_naive = lambda s, a, d, s2: (true_reward(s, a, d)
                                   + 1.0 * min(s + a, d)
                                   - 2.0 * max(d - s - a, 0))
    Phi = lambda s: -HOLD * s
    r_pot = lambda s, a, d, s2: (true_reward(s, a, d)
                                 + GAMMA * Phi(s2) - Phi(s))
    rows = []
    for name, rf, color in (("sparse", r_sparse, "#e6550d"),
                            ("naive shaped", r_naive, "#756bb1"),
                            ("potential", r_pot, "#31a354")):
        Q = run_qlearning(rf)
        perf = evaluate(Q)
        rows.append((name, perf, color))
        print(f"{name}: {perf:.2f}")
    print(f"training took {time.time()-t0:.0f}s")

    # exact optimum reference
    P = np.zeros((MAX_S + 1, MAX_S + 1, MAX_S + 1))
    R = np.zeros((MAX_S + 1, MAX_S + 1))
    for s in range(MAX_S + 1):
        for a in range(MAX_S - s + 1):
            R[s, a] = sum(pr * true_reward(s, a, d)
                          for d, pr in enumerate(PMF))
            for d, pr in enumerate(PMF):
                P[s, a, max(0, s + a - d)] += pr
    V = np.zeros(MAX_S + 1)
    for _ in range(500):
        Vn = (R + GAMMA * np.einsum("sap,p->sa", P, V)).max(axis=1)
        if np.abs(Vn - V).max() < 1e-10:
            V = Vn
            break
        V = Vn
    pol_star = (R + GAMMA * np.einsum("sap,p->sa", P, V)).argmax(axis=1)

    ax = axes[1]
    names = [r[0] for r in rows]
    perfs = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    bars = ax.bar(names, perfs, color=colors, width=0.55)
    rng = np.random.default_rng(99)
    s, tot = 0, 0.0
    for _ in range(8000):
        s = int(rng.integers(0, MAX_S + 1))
        g, G = 1.0, 0.0
        for _ in range(30):
            a = min(int(pol_star[s]), MAX_S - s)
            d = int(rng.choice(MAX_S + 1, p=PMF))
            G += g * true_reward(s, a, d)
            g *= GAMMA
            s = max(0, s + a - d)
        tot += G
    exact = tot / 8000
    ax.axhline(exact, color="black", lw=1.0, ls="--")
    ax.annotate(f"exact optimum {exact:.1f}", xy=(1.0, exact),
                xytext=(-130, 6), textcoords="offset points", fontsize=8)
    for b, p in zip(bars, perfs):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.6, f"{p:.1f}",
                ha="center", fontsize=9)
    ax.set_ylabel("TRUE performance (8000 rollouts)")
    ax.set_title("Reward design: only potential-based is safe")
    ax.set_ylim(min(perfs) - 8, exact + 6)

    assert max(perfs, key=lambda x: x) <= exact + 2.0, \
        "no trained policy should beat the exact optimum (MC noise 2)"
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")
    print("data:", {"terminal": {k: [int(pol_t[0]) for pol_t in
                                     []]} if False else "see left",
                     "reward": [(n, round(p, 2)) for n, p, _ in rows],
                     "exact": round(exact, 2)})


if __name__ == "__main__":
    main()
