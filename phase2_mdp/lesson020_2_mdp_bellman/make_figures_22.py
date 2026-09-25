"""Lesson 2.2 figure: Bellman machinery made visible.

Same inventory MDP as phase2_mdp/lesson2_2_mdp_inventory.py.
Left: VI convergence — max|V_k − V*| per sweep (contraction = geometric).
Right: the discovered optimal policy as order-amount bars per inventory
level (the (s,S) structure: order 4−s when s ≤ 4, else 0) + V* curve.
Run:  python phase2_mdp/make_figures_22.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "assets", "phase2")
os.makedirs(OUT, exist_ok=True)

MAX_INV = 10
LAMBDA = 3.0
PRICE, HOLD, COST = 10.0, 0.5, 3.0
GAMMA = 0.95
NS, NA = MAX_INV + 1, MAX_INV + 1


def build():
    pmf = poisson.pmf(np.arange(MAX_INV + 1), LAMBDA)
    pmf[-1] += 1.0 - pmf.sum()
    P = np.zeros((NS, NA, NS))
    R = np.full((NS, NA), -1e9)
    for s in range(NS):
        for a in range(NA):
            if a > MAX_INV - s:
                continue
            R[s, a] = sum(pr * (PRICE * min(s + a, d)
                                - HOLD * max(s + a - d, 0) - COST * a)
                          for d, pr in enumerate(pmf))
            for d, pr in enumerate(pmf):
                P[s, a, max(0, s + a - d)] += pr
    return P, R


def main():
    P, R = build()

    # VI with contraction tracking — run to full convergence (γ=0.95 needs
    # ~460 sweeps for 1e-9; the tail IS the geometric decay the plot shows)
    V = np.zeros(NS)
    errs, policy_hist = [], []
    for it in range(600):
        Q = R + GAMMA * np.einsum("sap,p->sa", P, V)
        Vn = Q.max(axis=1)
        errs.append(np.abs(Vn - V).max())
        policy_hist.append(Q.argmax(axis=1).copy())
        V = Vn
        if errs[-1] < 1e-9 and it > 50:
            break
    errs = np.array(errs)

    # final exact V and policy
    Q = R + GAMMA * np.einsum("sap,p->sa", P, V)
    pol = Q.argmax(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
    ax = axes[0]
    ax.semilogy(range(1, len(errs) + 1), np.maximum(errs, 1e-12), lw=1.6,
                color="#08519c",
                label="max |V_k − V_{k−1}| (contraction)")
    # ratio ~ gamma after transient
    k_geo = np.arange(5, len(errs) + 1)
    ax.semilogy(k_geo, errs[4] * GAMMA ** (k_geo - 5), ls=":", lw=1.2,
                color="gray", label=f"geometric decay ×γ = {GAMMA}")
    ax.set_xlabel("value-iteration sweep k")
    ax.set_ylabel("error (log)")
    ax.set_title(f"Bellman operator is a γ-contraction — VI converges "
                 f"geometrically ({len(errs)} sweeps)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.bar(np.arange(NS) - 0.2, pol, width=0.4, color="#31a354",
           alpha=0.9, label="optimal order amount π*(s)")
    ax2 = ax.twinx()
    ax2.plot(np.arange(NS), V, "o-", color="#756bb1", lw=1.5, ms=4,
             label="V*(s)")
    ax.axvspan(-0.5, 4.2, color="#fb6a4a", alpha=0.10)
    ax.text(1.8, 4.1, "(s,S) zone:\norder up to 5", fontsize=9,
            ha="center", color="#a50f15")
    ax.text(7.5, 4.1, "no re-order", fontsize=9, ha="center",
            color="dimgray")
    ax.set_xlabel("inventory level s")
    ax.set_ylabel("order amount")
    ax2.set_ylabel("V*(s)")
    ax.set_ylim(0, 5.2)
    ax.set_title("the solver discovered the business rule")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(OUT, "lesson2_2_bellman.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))
    print("policy:", list(pol))
    print("V(0):", round(float(V[0]), 2), "(lesson evidence: 373.40)")


if __name__ == "__main__":
    main()
