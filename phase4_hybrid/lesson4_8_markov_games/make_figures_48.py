"""Lesson 4.8 figure: Shapley value iteration convergence + equilibrium.

Same seeded pursuit game as phase4_hybrid/lesson4_8_markov_games.py.
Left: V estimates over VI iterations (all gaps) — the max-min nesting
still contracts; watch every curve settle by ~iteration 90.
Right: the game's value ladder V(gap) with the zero crossing annotated
(pursuit beyond gap 3 has negative value).
Run:  python phase2_mdp/make_figures_48.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "assets", "phase4")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "phase4_hybrid"))

from lesson4_8_markov_games import shapley_vi  # noqa: E402


def main():
    V, hist, it = shapley_vi()
    print(f"VI converged in {it} iterations; V = {np.round(V, 3)}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    ax = axes[0]
    colors = ["#08519c", "#31a354", "#d94801", "#756bb1", "#636363"]
    for g in range(1, 5):
        ax.plot(hist[:, g], lw=1.4, color=colors[g],
                label=f"gap {g} (V* = {V[g]:.2f})")
    ax.axhline(10.0, color="black", ls=":", lw=1.1, label="gap 0: 10 (annuity)")
    ax.set_xlabel("value-iteration step")
    ax.set_ylabel("V estimate")
    ax.set_title(f"Shapley VI convergence ({it} iterations)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    gaps = np.arange(5)
    bar_colors = ["#31a354" if v >= 0 else "#d9534f" for v in V]
    ax.bar(gaps, V, color=bar_colors, alpha=0.85)
    for g, v in zip(gaps, V):
        ax.text(g, v + (0.25 if v >= 0 else -0.55), f"{v:.2f}",
                ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=1)
    ax.annotate("zero crossing: pursuit beyond\n"
                "gap 3 has negative value",
                xy=(3, -0.433), xytext=(2.1, 4.5), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.set_xticks(gaps)
    ax.set_xlabel("gap (distance to evader)")
    ax.set_ylabel("game value V(gap) for the pursuer")
    ax.set_title("the value ladder — what the pursuit is worth")

    fig.tight_layout()
    path = os.path.join(OUT, "lesson4_8_shapley_vi.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
