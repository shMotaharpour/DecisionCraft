"""Lesson 1.10 figure: Branch-and-Price in numbers.

Left: best integer value found vs B&P node budget (the stall made visible)
against the direct-MILP optimum and the LP/CG bound lines.
Right: the five-method comparison as horizontal bars — the naive-rounding
INFEASIBLE bar rendered hatched and red.
Run:  python phase2_mdp/make_figures_110.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "assets", "phase1")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "phase1_milp"))

from lesson1_10_branch_and_price import (  # noqa: E402
    branch_and_price, dw_cg_bound, make_instance, naive_round,
    solve_direct_lp_relax, solve_direct_mip)


def main():
    blocks, gu, gc = make_instance()
    opt_mip, _, t_mip = solve_direct_mip(blocks, gu, gc)
    lp = solve_direct_lp_relax(blocks, gu, gc)
    master_val, cols, mu, pi, it = dw_cg_bound(blocks, gu, gc)
    val_r, feas, _ = naive_round(blocks, cols, gu, gc)

    # B&P best-value trajectory over node budgets (cumulative: reuse the
    # DFS at increasing budgets — the lesson's DFS restarts, so curve is
    # step-like; that step shape is itself the "stall" finding)
    budgets = [100, 200, 400, 800, 1600, 3200]
    bp_vals, bp_nodes = [], []
    for nb in budgets:
        val, nodes, t = branch_and_price(blocks, gu, gc, node_budget=nb)
        bp_vals.append(val)
        bp_nodes.append(nodes)
        print(f"budget {nb}: best {val:.2f} ({nodes} nodes, {t:.0f}s)")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    ax = axes[0]
    ax.plot(bp_nodes, bp_vals, "o-", color="#d94801", lw=1.5,
            label="branch-and-price best found")
    ax.axhline(opt_mip, color="#08519c", ls="-", lw=1.4,
               label=f"direct MILP optimum {opt_mip:.1f} (0.5 s)")
    ax.axhline(lp, color="gray", ls=":", lw=1.2,
               label=f"LP / CG bound {lp:.1f}")
    ax.set_xscale("log")
    ax.set_xlabel("node budget (log)")
    ax.set_ylabel("best integer value found")
    ax.set_title("toy B&P stalls: best value vs node budget")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    names = ["direct MILP\n(optimum)", "LP / CG\nbound",
             "naive rounding\nof λ mix", "branch-and-price\n(400 nodes)"]
    vals = [opt_mip, lp, val_r, bp_vals[2]]
    colors = ["#08519c", "#6baed6", "#d62728", "#d94801"]
    hatch = ["", "", "//", ""]
    bars = ax.barh(names, vals, color=colors, hatch=hatch, alpha=0.85,
                   height=0.55)
    for bar, v, feas in zip(bars, vals, [True, True, feas, True]):
        lbl = f"{v:.1f}" + ("" if feas else "  (INFEASIBLE)")
        ax.text(bar.get_width() + (2 if v >= 0 else -2),
                bar.get_y() + bar.get_height() / 2, lbl,
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.axvline(opt_mip, color="#08519c", ls="--", lw=1.1)
    ax.set_xlabel("profit (direct MILP optimum = dashed line)")
    ax.set_title("the rounding trap: 417.9 > optimum ⇒ infeasible")
    ax.set_xlim(min(390, val_r - 8), 428)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    path = os.path.join(OUT, "lesson1_10_branch_price.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
