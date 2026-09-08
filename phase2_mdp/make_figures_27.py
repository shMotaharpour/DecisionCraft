"""Lesson 2.7 figure: MC vs TD(0) convergence curves against exact V.

Same policy and dynamics as phase2_mdp/lesson2_7_mc_td_cmdp.py (the (s,S)
inventory policy). Two panels: value-estimate error over samples, and the
final V estimate vs exact on the visited support. Seeded.
Run:  python phase2_mdp/make_figures.py 2_7   (or 'all')
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "phase2")
os.makedirs(OUT, exist_ok=True)

MAX_INV = 10
LAMBDA = 3.0
PRICE, HOLD, COST = 10.0, 0.5, 3.0
GAMMA = 0.95
NS, NA_ = MAX_INV + 1, MAX_INV + 1


def build_model_and_policy():
    pmf = poisson.pmf(np.arange(MAX_INV + 1), LAMBDA)
    pmf[-1] += 1.0 - pmf.sum()
    P = np.zeros((NS, NA_, NS))
    R = np.full((NS, NA_), -1e9)
    for s in range(NS):
        for a in range(NA_):
            if a > MAX_INV - s:
                continue
            R[s, a] = sum(pr * (PRICE * min(s + a, d)
                                - HOLD * max(s + a - d, 0) - COST * a)
                          for d, pr in enumerate(pmf))
            for d, pr in enumerate(pmf):
                P[s, a, max(0, s + a - d)] += pr
    policy = np.zeros(NS, dtype=int)
    V = None
    for _ in range(100):
        P_pi = P[np.arange(NS), policy]
        R_pi = R[np.arange(NS), policy]
        V = np.linalg.solve(np.eye(NS) - GAMMA * P_pi, R_pi)
        new_pol = (R + GAMMA * np.einsum("sap,p->sa", P, V)).argmax(axis=1)
        if np.array_equal(new_pol, policy):
            break
        policy = new_pol
    return P, R, policy, V


def fig_mc_td():
    P, R, policy, V_exact = build_model_and_policy()
    P_pi = P[np.arange(NS), policy]
    R_pi = R[np.arange(NS), policy]
    rng = np.random.default_rng(7)

    STEPS = 1_000_000
    CHECK = np.unique(np.linspace(1, STEPS, 200).astype(int))
    traj_s = np.empty(STEPS, dtype=int)
    traj_r = np.empty(STEPS)
    s = 0
    for t in range(STEPS):
        traj_s[t] = s
        traj_r[t] = R_pi[s]
        s = rng.choice(NS, p=P_pi[s])
    G = np.empty(STEPS)
    G[-1] = traj_r[-1]
    for t in range(STEPS - 2, -1, -1):
        G[t] = traj_r[t] + GAMMA * G[t + 1]

    support = np.arange(6)          # states 0..5 are the visited support
    # ---- MC curve: error vs #returns consumed (first-visit, thinned)
    mc_err = []
    counts = np.zeros(NS)
    sums = np.zeros(NS)
    next_i = 0
    for c in CHECK:
        while next_i < c:
            st = traj_s[next_i]
            if st in support:
                sums[st] += G[next_i]
                counts[st] += 1
            next_i += 1
        v = sums / np.maximum(counts, 1)
        mc_err.append(np.abs(v - V_exact)[support].max())
    mc_err = np.array(mc_err)

    # ---- TD(0) curve: re-run online, record error at checkpoints
    V = np.zeros(NS)
    td_err = []
    s = 0
    alpha = 0.02
    ci = 0
    for t in range(STEPS):
        ns_ = rng.choice(NS, p=P_pi[s])
        V[s] += alpha * (R_pi[s] + GAMMA * V[ns_] - V[s])
        s = ns_
        if t + 1 == CHECK[ci]:
            td_err.append(np.abs(V - V_exact)[support].max())
            ci += 1
            if ci >= len(CHECK):
                break
    td_err = np.array(td_err)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    ax = axes[0]
    ax.plot(CHECK, mc_err, lw=1.5, color="#d94801",
            label="MC (first-visit returns)")
    ax.plot(CHECK[:len(td_err)], td_err, lw=1.5, color="#08519c",
            label="TD(0) bootstrapping, α=0.02")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("environment steps (log)")
    ax.set_ylabel("max |V̂ − V_exact| on visited states")
    ax.set_title("convergence: MC vs TD(0) — same data stream")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(support, V_exact[support], "o-", color="black", lw=1.4,
            label="exact V (linear solve)")
    ax.plot(support, (sums / np.maximum(counts, 1))[support], "s--",
            color="#d94801", ms=5, label="MC estimate (1M steps)")
    ax.plot(support, V[support], "^--", color="#08519c", ms=5,
            label="TD(0) estimate (1M steps)")
    ax.set_xlabel("inventory state s")
    ax.set_ylabel("value")
    ax.set_title("final estimates vs exact (states 6–10 never visited)")
    ax.axvspan(5.5, 10.4, color="gray", alpha=0.12)
    ax.text(7.9, V_exact.min() + 6, "zero visits\n(on-policy support)",
            fontsize=8, ha="center", color="dimgray")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, "lesson2_7_mc_td.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))


if __name__ == "__main__":
    fig_mc_td()
