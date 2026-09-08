"""Generate course figures: regret curves for lesson 3.5 bandit tournament.

Outputs assets/phase3/lesson3_5_regret.png (committed to the repo and
embedded in notes/phase3_lesson5_bandits.md). Reruns the same seeded
simulation as the lesson so figure and evidence stay consistent.
Run:  python phase3_rl/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "phase3")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(11)
K = 10
MU = np.sort(rng.uniform(0.2, 0.8, K))[::-1]
MU_STAR = MU[0]
T = 40_000


def pull(i):
    return float(rng.random() < MU[i])


def run_eps(eps_fn):
    Q, N, regret = np.zeros(K), np.zeros(K), np.zeros(T)
    for t in range(T):
        a = rng.integers(K) if rng.random() < eps_fn(t) else int(Q.argmax())
        r = pull(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        regret[t] = regret[t - 1] + (MU_STAR - MU[a])
    return regret


def run_ucb1(c=2.0):
    Q, N, regret = np.zeros(K), np.zeros(K), np.zeros(T)
    for t in range(T):
        a = t if t < K else int((Q + c * np.sqrt(np.log(t) / N)).argmax())
        r = pull(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        regret[t] = regret[t - 1] + (MU_STAR - MU[a])
    return regret


def run_thompson():
    alpha, beta = np.ones(K), np.ones(K)   # NOT alpha = beta = ones (aliasing!)
    regret = np.zeros(T)
    for t in range(T):
        a = int(rng.beta(alpha, beta).argmax())
        r = pull(a)
        alpha[a] += r
        beta[a] += 1 - r
        regret[t] = regret[t - 1] + (MU_STAR - MU[a])
    return regret


def main():
    curves = {
        "ε constant 0.10 (linear regret)": run_eps(lambda t: 0.10),
        "ε decay 1/√t": run_eps(lambda t: min(1.0, 1 / np.sqrt(t + 1))),
        "UCB1 c=2 (proven log regret)": run_ucb1(),
        "Thompson sampling": run_thompson(),
    }
    colors = {"ε constant 0.10 (linear regret)": "#d62728",
              "ε decay 1/√t": "#ff7f0e",
              "UCB1 c=2 (proven log regret)": "#1f77b4",
              "Thompson sampling": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    ax = axes[0]
    for name, reg in curves.items():
        ax.plot(reg, label=name, color=colors[name], lw=1.4)
    ax.set_xlabel("step t")
    ax.set_ylabel("cumulative regret")
    ax.set_title(f"Bandit regret — {K} arms, T={T:,} (seed 11)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    marks = np.array([100, 300, 1_000, 3_000, 10_000, 40_000])
    for name, reg in curves.items():
        ax.plot(marks, reg[marks - 1], "o-", color=colors[name], lw=1.4)
    ax.plot(marks, 0.013 * marks, "--", color="gray", lw=1,
            label="linear growth ∝ t")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step t (log)")
    ax.set_ylabel("regret at t (log)")
    ax.set_title("log-log view: linear vs logarithmic growth")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(OUT, "lesson3_5_regret.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, os.path.join(HERE, "..")))


if __name__ == "__main__":
    main()
