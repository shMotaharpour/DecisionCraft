"""Lesson 2.5 figure: mean first-passage matrix as a heatmap + Kemeny check.

Same market chain as phase2_mdp/lesson2_5_absorbing_fit.py part B.
Left: mean first-passage time matrix M (rows = from, cols = to), computed
by first-step linear systems and annotated. Right: mean RETURN times
1/pi_i (Kemeny's law) as bars — the diagonal's complement.
Run:  python phase2_mdp/make_figures_25b.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "phase2")
os.makedirs(OUT, exist_ok=True)

P = np.array([[0.85, 0.10, 0.05],
              [0.20, 0.70, 0.10],
              [0.30, 0.30, 0.40]])
n = 3
NAMES = ["bull", "bear", "stagnant"]

# MFPT by first-step analysis (verified vs MC in the lesson's evidence)
M = np.zeros((n, n))
for j in range(n):
    idx = [i for i in range(n) if i != j]
    M[idx, j] = np.linalg.solve(
        np.eye(n - 1) - P[np.ix_(idx, idx)], np.ones(n - 1))

evals, evecs = np.linalg.eig(P.T)
k = np.argmin(np.abs(evals - 1.0))
pi = np.real(evecs[:, k])
pi /= pi.sum()


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)

    ax = axes[0]
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto")
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        fontsize=10,
                        color="white" if M[i, j] > M.max() * 0.6 else
                        "black")
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=10, color="#bbbbbb")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(NAMES); ax.set_yticklabels(NAMES)
    ax.set_xlabel("target state (to)")
    ax.set_ylabel("start state (from)")
    ax.set_title("mean first-passage times (months)")
    fig.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[1]
    ret = 1.0 / pi
    bars = ax.bar(NAMES, ret, color=["#31a354", "#d94801", "#756bb1"],
                  alpha=0.85)
    for bar, v in zip(bars, ret):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.12, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.set_ylabel("mean return time  1/πᵢ  (months)")
    ax.set_title(f"Kemeny's law: return time = 1/πᵢ  "
                 f"(π = {', '.join(f'{p:.2f}' for p in pi)})")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(OUT, "lesson2_5_mfp.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))
    print("M =\n", np.round(M, 2))
    print("1/pi =", np.round(ret, 2))


if __name__ == "__main__":
    main()
