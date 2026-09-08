"""Lesson 2.1 figure: the market-regime chain.

Left: state-transition graph with labeled edges (matplotlib, no networkx).
Right: convergence of the state distribution to the stationary one —
analytic π* vs 100k-month simulation frequencies (the lesson's own numbers).
Run:  python phase2_mdp/make_figures_21.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "phase2")
os.makedirs(OUT, exist_ok=True)

STATES = ["bull", "bear", "stagnant"]
P = np.array([[0.85, 0.10, 0.05],
              [0.20, 0.70, 0.10],
              [0.30, 0.30, 0.40]])
POS = {"bull": (0.9, 0.78), "bear": (0.15, 0.72), "stagnant": (0.5, 0.18)}


def main():
    evals, evecs = np.linalg.eig(P.T)
    k = np.argmin(np.abs(evals - 1.0))
    pi_star = np.real(evecs[:, k])
    pi_star /= pi_star.sum()

    # simulation long-run frequencies (same seed/length as the lesson)
    rng = np.random.default_rng(42)
    N = 100_000
    state = 1
    visits = np.zeros(3)
    for _ in range(N):
        visits[state] += 1
        state = rng.choice(3, p=P[state])
    sim_freq = visits / N

    fig = plt.figure(figsize=(11, 4.4), dpi=150)
    axg = fig.add_subplot(1, 2, 1)
    axg.set_xlim(0, 1.05)
    axg.set_ylim(0, 1)
    axg.axis("off")
    axg.set_title("market-regime chain (monthly transitions)")

    # edges (self-loops drawn as circles, cross edges as arrows)
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            pa, pb = np.array(POS[STATES[a]]), np.array(POS[STATES[b]])
            mid = (pa + pb) / 2
            # curve the arrows by offsetting control point
            off = np.array([-(pb[1] - pa[1]), pb[0] - pa[0]]) * 0.09
            axg.annotate("", xy=pb - (pb - pa) * 0.09,
                         xytext=pa + (pb - pa) * 0.09,
                         arrowprops=dict(arrowstyle="->", lw=1.2,
                                         color="#555555",
                                         connectionstyle="arc3,rad=0.25"))
            axg.text(*(mid + off * 1.6), f"{P[a, b]:.2f}", fontsize=8,
                     ha="center", color="#555555")
    for name, (x, y) in POS.items():
        idx = STATES.index(name)
        share = pi_star[idx]
        axg.add_patch(plt.Circle((x, y), 0.085, color="#bdd7ee",
                                 ec="#08519c", lw=1.6, zorder=3))
        axg.text(x, y + 0.012, name, ha="center", fontsize=10, zorder=4)
        axg.text(x, y - 0.030, f"π*={share:.2f}", ha="center", fontsize=8,
                 zorder=4)
    # self-loop labels
    for name, lbl in (("bull", 0.85), ("bear", 0.70), ("stagnant", 0.40)):
        x, y = POS[name]
        axg.add_patch(plt.Circle((x, y + 0.135), 0.045, fill=False,
                                 ec="#999999", lw=1.0, zorder=2))
        axg.text(x, y + 0.205, f"{lbl:.2f}", ha="center", fontsize=8,
                 color="#777777")

    axc = fig.add_subplot(1, 2, 2)
    ks = np.arange(0, 13)
    mats = np.array([np.linalg.matrix_power(P, k) for k in ks])
    # distribution after k steps starting from bear
    dists = mats @ np.array([0.0, 1.0, 0.0])
    for j, name in enumerate(STATES):
        axc.plot(ks, dists[:, j], "o-", ms=4, lw=1.4,
                 label=f"{name} (π*={pi_star[j]:.2f})")
        axc.axhline(pi_star[j], color=axc.lines[-1].get_color(), ls=":",
                    lw=1)
    axc.set_xlabel("months ahead k (starting from bear)")
    axc.set_ylabel("P(state at t+k)")
    axc.set_title("k-step forecasts converge to π* (dotted)")
    axc.legend(fontsize=8)
    axc.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, "lesson2_1_chain.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))
    print("pi*:", np.round(pi_star, 4), "| sim:", np.round(sim_freq, 4))


if __name__ == "__main__":
    main()
