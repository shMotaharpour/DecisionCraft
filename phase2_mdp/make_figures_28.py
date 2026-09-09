"""Lesson 2.8 figure: the two resolution dials, measured.

Left  — time dial: optimal value vs clock granularity (epochs 3..21 on
the 42-day season), monotone non-decreasing with a visible plateau, and
|S| growth on the same axes (twin).
Right — state dial: lifted-rollout performance vs number of state bins
(3..21), with the exact model marked; the flat region tolerates binning,
the bend near the reorder structure does not.

Run: python phase2_mdp/make_figures_28.py   (~60 s, seeded)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesson2_8a_time_granularity import (DAYS, MAX_STOCK, solve_clock,
                                        START_STOCK)
from lesson2_8b_state_granularity import lift_and_evaluate, solve

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "phase2",
                   "lesson2_8_resolution.png")


def main():
    # ---- left: time dial sweep ----
    epochs_list = [3, 4, 6, 7, 14, 21]
    days_per_ok = [e for e in epochs_list if DAYS % e == 0]
    vals_t, sizes_t = [], []
    for e in days_per_ok:
        V, pol, n_states = solve_clock(e)
        vals_t.append(V[0, START_STOCK])
        sizes_t.append(n_states)
        print(f"clock epochs={e:2d}: V={V[0, START_STOCK]:.2f} |S|={n_states}")

    # ---- right: state dial sweep ----
    step_list = [10, 5, 3, 2, 1]
    bins_s, vals_s = [], []
    for step in step_list:
        states = np.arange(0, MAX_STOCK + 1, step)
        amax = [MAX_STOCK - s for s in states]
        V, pol, _ = solve(states, amax)
        perf = lift_and_evaluate(pol, states)
        bins_s.append(len(states))
        vals_s.append(perf)
        print(f"state bins={len(states):2d} (step {step:2d}): "
              f"true perf={perf:.2f}")

    # asserts: theory promises monotone-in-fineness on both dials
    assert all(b >= a - 1e-6 for a, b in zip(vals_t, vals_t[1:])), \
        "time-dial values must be non-decreasing in fineness"
    assert all(b >= a - 3.0 for a, b in zip(vals_s, vals_s[1:])), \
        "state-dial lifted perf should improve toward exact (MC noise 3)"

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    ax = axes[0]
    ax.plot(days_per_ok, vals_t, "o-", color="#08519c", lw=1.8, ms=6,
            label="optimal value V(s=15)")
    ax.set_xlabel("decision epochs in the 42-day season (coarse → fine)")
    ax.set_ylabel("optimal value", color="#08519c")
    ax2 = ax.twinx()
    ax2.plot(days_per_ok, sizes_t, "s--", color="#e6550d", lw=1.4, ms=5,
             label="state-space size |S|")
    ax2.set_ylabel("|S| (model cost)", color="#e6550d")
    ax.set_title("Time granularity: value rises, cost rises with it")
    ax.legend(loc="lower right", fontsize=8)
    ax2.legend(loc="center right", fontsize=8)

    ax = axes[1]
    ax.plot(bins_s, vals_s, "o-", color="#31a354", lw=1.8, ms=6)
    ax.axhline(vals_s[-1], color="gray", lw=0.8, ls=":")
    ax.annotate("exact model", xy=(bins_s[-1], vals_s[-1]),
                xytext=(-78, -16), textcoords="offset points", fontsize=8,
                color="gray")
    ax.set_xlabel("number of state bins (coarse → fine)")
    ax.set_ylabel("TRUE performance (lifted rollouts)")
    ax.set_ylim(vals_s[0] - 1.2, vals_s[-1] + 1.2)
    ax.set_title("State granularity: bin where V is flat")
    loss = 100 * (vals_s[-1] - vals_s[0]) / abs(vals_s[-1])
    ax.annotate(f"coarsest bins: −{loss:.1f}%",
                xy=(bins_s[0], vals_s[0]),
                xytext=(14, 14), textcoords="offset points", fontsize=9,
                color="#31a354")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")
    print("data:",
          {"time": list(zip(days_per_ok, [round(v, 2) for v in vals_t],
                             sizes_t)),
           "state": list(zip(bins_s, [round(v, 2) for v in vals_s]))})


if __name__ == "__main__":
    main()
