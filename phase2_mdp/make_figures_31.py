"""Lesson 3.1 figure: tabular RL learning curves (MC/SARSA/Q-learning).

Re-runs the three lesson agents (same seeds) and plots smoothed training
returns vs the exact (s,S) policy's evaluated performance line.
Runtime ~3-4 min (three 60k-episode runs).
Run:  python phase2_mdp/make_figures_31.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "assets", "phase3")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "phase3_rl"))

import lesson3_1_qlearning_inventory as L  # noqa: E402


def main():
    V_exact, policy_exact = L.exact_solution()
    exact_perf = L.evaluate(policy_exact)
    print(f"exact (s,S) performance: {exact_perf:.1f}")

    curves = {}
    print("training MC ...")
    _, c = L.train_mc(seed=7)
    curves["MC (full-episode returns)"] = np.array(c)
    print("training SARSA ...")
    _, c = L.train_td("sarsa", seed=7)
    curves["SARSA (on-policy TD)"] = np.array(c)
    print("training Q-learning ...")
    Q, c = L.train_td("qlearn", seed=7)
    curves["Q-learning (off-policy TD)"] = np.array(c)

    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=150)
    colors = {"MC (full-episode returns)": "#d94801",
              "SARSA (on-policy TD)": "#31a354",
              "Q-learning (off-policy TD)": "#08519c"}
    K = 2000
    for name, c in curves.items():
        sm = np.convolve(c, np.ones(K) / K, mode="valid")
        ax.plot(sm, lw=1.3, color=colors[name], label=name)
    ax.axhline(exact_perf, color="black", ls="--", lw=1.3,
               label=f"exact (s,S) policy: {exact_perf:.0f}")
    ax.set_xlabel("episode")
    ax.set_ylabel(f"training return (moving average, k={K})")
    ax.set_title("tabular RL learning curves — model-blind inventory (seed 7)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, "lesson3_1_learning_curves.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
