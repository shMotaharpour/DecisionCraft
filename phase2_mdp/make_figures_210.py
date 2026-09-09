"""Lesson 2.10 figure: constraint pricing + deployment dial.

Left — mu landscape: violation rate and TRUE profit vs penalty mu.
The violation rate JUMPS at mu* (discrete actions); profit falls from
the law-breaking 300 to the feasible 246 as mu crosses it. mu* is the
budget's shadow price.
Right — deployment: cumulative profit, static policy vs MPC across
the 30-day horizon with the day-15 demand shift marked.

Run: python phase2_mdp/make_figures_210.py   (~3 min, seeded)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesson2_10a_constraint_penalty import (BUDGET, COST, build_P_R,
                                           rollout, vi_policy)
from lesson2_10b_precompute_mpc import (DAYS, LAM_POST, LAM_PRE, N_SEEDS,
                                       SHIFT_DAY, mpc_action, pmf_for,
                                       solve_static)

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "phase2",
                   "lesson2_10_constraint_mpc.png")


def main():
    # ---- left: mu landscape ----
    mus = [0.0, 0.5, 1.0, 1.5, 2.0, 2.2, 2.4, 2.6, 3.0, 4.0, 6.0, 10.0]
    viols, profits = [], []
    for mu in mus:
        pol = vi_policy(*build_P_R(pen_mu=mu))
        perf, viol, spend = rollout(pol, n=12000)
        viols.append(100 * viol)
        profits.append(perf)
        print(f"mu={mu:4.1f}: viol={100*viol:6.2f}% profit={perf:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    ax = axes[0]
    ax.plot(mus, viols, "o-", color="#e6550d", lw=1.8, ms=5,
            label="budget violations")
    ax.set_xlabel("penalty mu (per unit of overspend)")
    ax.set_ylabel("violation rate (%)", color="#e6550d")
    ax2 = ax.twinx()
    ax2.plot(mus, profits, "s-", color="#08519c", lw=1.6, ms=5,
             label="TRUE profit")
    ax2.set_ylabel("TRUE profit", color="#08519c")
    jump_i = next(i for i in range(1, len(mus))
                  if viols[i] < 0.5 <= viols[i - 1])
    mu_star = (mus[jump_i - 1] + mus[jump_i]) / 2
    ax.axvline(mu_star, color="black", lw=1.0, ls="--")
    ax.annotate(f"mu* shadow price\n(~{mu_star:.2f})", xy=(mu_star, 20),
                xytext=(10, 0), textcoords="offset points", fontsize=8)
    ax.set_title("Penalty pricing: violations jump at mu*")
    ax.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="center right", fontsize=8)

    # ---- right: deployment race, cumulative profit (mean over seeds) ----
    all_cum = {"static": [], "mpc": []}
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        static_pol = solve_static((LAM_PRE + LAM_POST) / 2)
        lam_runs, sales_today = LAM_PRE, 0.0
        cum = {"static": [0.0], "mpc": [0.0]}
        s = {"static": 0, "mpc": 0}
        for t in range(DAYS):
            lam_true = LAM_PRE if t < SHIFT_DAY else LAM_POST
            for mode in ("static", "mpc"):
                if mode == "mpc":
                    a = mpc_action(s[mode], t, lam_runs)
                else:
                    a = int(static_pol[s[mode]])
                a = min(a, 12 - s[mode])
                d = int(rng.choice(13, p=pmf_for(lam_true)))
                sold = min(s[mode] + a, d)
                r = 10.0 * sold - 0.4 * max(s[mode] + a - d, 0) - 3.0 * a
                cum[mode].append(cum[mode][-1] + r)
                s[mode] = max(0, s[mode] + a - d)
            sales_today = 0.5 * sales_today + 0.5 * d
            lam_runs = max(0.3, sales_today)
        for mode in ("static", "mpc"):
            all_cum[mode].append(cum[mode])
    mean = {m: np.mean(all_cum[m], axis=0) for m in all_cum}
    lo = {m: np.percentile(all_cum[m], 15, axis=0) for m in all_cum}
    hi = {m: np.percentile(all_cum[m], 85, axis=0) for m in all_cum}

    ax = axes[1]
    days = range(DAYS + 1)
    ax.plot(days, mean["static"], color="#756bb1", lw=1.8,
            label="static VI policy (avg-lambda model)")
    ax.fill_between(days, lo["static"], hi["static"],
                    color="#756bb1", alpha=0.15)
    ax.plot(days, mean["mpc"], color="#31a354", lw=1.8,
            label="MPC (H=5, online lambda)")
    ax.fill_between(days, lo["mpc"], hi["mpc"], color="#31a354", alpha=0.15)
    ax.axvline(SHIFT_DAY, color="black", lw=1.0, ls="--")
    ax.annotate(f"demand {LAM_PRE}->{LAM_POST}\n(promotion)",
                xy=(SHIFT_DAY, 60),
                xytext=(8, -6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("day")
    ax.set_ylabel("cumulative profit (mean of 24 seeds, 15-85% band)")
    ax.set_title("MPC rides the shift the static model cannot see")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")
    print("data:", {"mu": mus, "viol": [round(v, 2) for v in viols],
                    "profit": [round(p, 2) for p in profits],
                    "mu_star": round(mu_star, 2),
                    "final_static_mean": round(float(mean["static"][-1]), 2),
                    "final_mpc_mean": round(float(mean["mpc"][-1]), 2)})


if __name__ == "__main__":
    main()
