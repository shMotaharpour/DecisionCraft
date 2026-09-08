"""Phase-1 lesson 1.1 figure: LP geometry — the picture under everything.

The lesson 1.1 toy portfolio in 2D (asset A vs B):
  max 15x + 12y  s.t.  x + y <= 10 (budget), x <= 6 (exposure), y >= 2
Left: feasible region shaded, constraints as lines, the optimal vertex
marked, iso-profit lines swept — "the optimum is a vertex" made visible.
Right: profit along the budget edge showing why the optimum sits at a
constraint intersection, not inside.
Run:  python phase1_milp/make_figures_11.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "phase1")
os.makedirs(OUT, exist_ok=True)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    ax = axes[0]
    x = np.linspace(0, 8, 400)
    y_budget = 10 - x                 # x + y <= 10
    # feasible region polygon: (0,2)-(4,2)-(4,6)-(6,4)-(6,2)-(0,2)
    poly_x = [0, 4, 4, 6, 6, 0]
    poly_y = [2, 2, 6, 4, 2, 2]
    ax.fill(poly_x, poly_y, color="#bdd7ee", alpha=0.75, zorder=1,
            label="feasible region")
    ax.plot(x, y_budget, color="#08519c", lw=1.5, label="x + y ≤ 10 (budget)")
    ax.axvline(6, color="#d94801", lw=1.5, label="x ≤ 6 (exposure)")
    ax.axhline(2, color="#31a354", lw=1.5, label="y ≥ 2 (min holding)")
    # iso-profit lines 15x + 12y = P for several P
    for P in (60, 90, 132):
        yy = (P - 15 * x) / 12
        ax.plot(x, yy, ls="--", lw=0.9, color="#999999", zorder=1)
        ax.text(0.1, (P) / 12 - 0.15, f"P={P}", fontsize=7, color="#777777",
                va="top")
    # optimal vertex: corner (4,6) from budget+? compute: candidates
    # (4,6): 15*4+12*6 = 132; (6,4): 138? but x<=6, y=10-6=4: 90+48=138.
    # Check (6,4) feasibility: x+y=10 ok, y>=2 ok. So optimum is (6,4)!
    # (lesson's numbers: 15/12 per unit; verify directly)
    for (vx, vy) in [(4, 6), (6, 4), (6, 2), (0, 2), (0, 10)]:
        if vy <= 10 - vx and vy >= 2 and vx <= 6:
            val = 15 * vx + 12 * vy
            ax.plot(vx, vy, "o", ms=7, color="black", zorder=5)
            ax.annotate(f"({vx},{vy}) → {val}", (vx, vy),
                        textcoords="offset points", xytext=(8, 5),
                        fontsize=8)
    ax.annotate("optimal vertex", (6, 4), textcoords="offset points",
                xytext=(10, -14), fontsize=9, color="#a50f15",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#a50f15"))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 11)
    ax.set_xlabel("asset A units (x)")
    ax.set_ylabel("asset B units (y)")
    ax.set_title("LP geometry: optimum = a vertex of the feasible region")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    # profit along the budget edge parametrized by x in [4..6]: y = 10-x
    xs = np.linspace(4, 6, 200)
    prof_edge = 15 * xs + 12 * (10 - xs)          # 120 + 3x → max at x=6
    ax.plot(xs, prof_edge, lw=1.8, color="#08519c",
            label="profit along budget edge (y = 10 − x)")
    xs2 = np.linspace(0, 6, 200)
    prof_x6 = 15 * 6 + 12 * np.clip(10 - 6, 2, 4) * 0 + 12 * np.where(
        xs2 + 4 <= 10, 4, np.nan)  # vertical edge x=6, y in [2,4]
    prof_x6 = 90 + 12 * np.linspace(2, 4, 200)
    ax.plot(np.full(200, 6), prof_x6, lw=1.8, color="#d94801",
            label="profit along exposure edge (x = 6)")
    ax.scatter([6], [138], color="black", zorder=5, s=45)
    ax.annotate("vertex (6,4): the gradient (15,12) points\n"
                "out of the region; the last supporting\n"
                "constraint is x ≤ 6",
                (6, 138), textcoords="offset points", xytext=(-165, -40),
                fontsize=8, va="top")
    ax.set_xlabel("x (asset A)")
    ax.set_ylabel("profit 15x + 12y")
    ax.set_title("profit is linear → maxima live on vertices/edges")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, "lesson1_1_lp_geometry.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
