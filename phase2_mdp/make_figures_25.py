"""Lesson 2.5 figure: gambler's ruin — ruin probability + time to absorption.

Same model as phase2_mdp/lesson2_5_absorbing_fit.py part A: stakes 0..6,
p(win a step) = 0.45. Two panels:
  left  — ruin probability from the fundamental matrix B vs closed form
  right — expected time to absorption (peaks mid-range: the non-obvious
          finding quoted in the notes)
Run:  python phase2_mdp/make_figures_25.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "phase2")
os.makedirs(OUT, exist_ok=True)

N_STAKE = 6
P_WIN = 0.45

# fundamental matrix (same construction as the lesson script)
P_full = np.zeros((N_STAKE + 1, N_STAKE + 1))
P_full[0, 0] = P_full[N_STAKE, N_STAKE] = 1.0
for s in range(1, N_STAKE):
    P_full[s, s - 1] = 1 - P_WIN
    P_full[s, s + 1] = P_WIN
Q = P_full[1:N_STAKE, 1:N_STAKE]
R = P_full[1:N_STAKE, [0, N_STAKE]]
Nmat = np.linalg.inv(np.eye(N_STAKE - 1) - Q)
B = Nmat @ R
t_absorb = Nmat @ np.ones(N_STAKE - 1)

# closed forms (proven)
i = np.arange(1, N_STAKE)
rho = (1 - P_WIN) / P_WIN
ruin_closed = (rho ** i - rho ** N_STAKE) / (1 - rho ** N_STAKE)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

    ax = axes[0]
    ax.plot(i, B[:, 0], "o-", color="#d62728", lw=1.6, ms=5,
            label="fundamental matrix N=(I−Q)⁻¹")
    ax.plot(i, ruin_closed, "x--", color="black", lw=1.1, ms=6,
            label="closed form (ρⁱ−ρᴺ)/(1−ρᴺ)")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.text(1.05, 0.515, "50/50 line", fontsize=8, color="gray")
    ax.set_xlabel("starting stake i")
    ax.set_ylabel("probability of ruin")
    ax.set_title(f"ruin probability — p(win step) = {P_WIN}")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(i, t_absorb, "o-", color="#08519c", lw=1.6, ms=5)
    imax = int(t_absorb.argmax())
    ax.annotate(f"peak at stake {i[imax]}: {t_absorb[imax]:.2f} steps\n"
                "(the game lasts LONGEST in the middle)",
                xy=(i[imax], t_absorb[imax]),
                xytext=(i[imax] + 0.4, t_absorb[imax] - 2.2),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.set_xlabel("starting stake i")
    ax.set_ylabel("E[steps to ruin or target]")
    ax.set_title("expected time to absorption — peaks mid-range")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, "lesson2_5_ruin.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))
    print("B[:,0] =", np.round(B[:, 0], 3))
    print("t_absorb =", np.round(t_absorb, 2))


if __name__ == "__main__":
    main()
