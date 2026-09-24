"""Phase-3 lesson 3.5 figure generator (split from the old multi-lesson
make_figures.py during the lesson-centric restructure). Generates the
bandit regret curves: assets/phase3/lesson3_5_regret.png."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "assets", "phase3", "lesson3_5_regret.png")

# Same setup as lesson3_5_bandits.py: 10-arm bandit, seeded
rng = np.random.default_rng(7)
K, T = 10, 20000
TRUE_P = rng.uniform(0.1, 0.9, K)
BEST = TRUE_P.max()


def run_epsilon(eps):
    Q = np.zeros(K)
    counts = np.zeros(K)
    regret = np.zeros(T)
    total_reg = 0.0
    for t in range(T):
        if rng.random() < eps:
            a = rng.integers(K)
        else:
            a = int(Q.argmax())
        r = 1.0 if rng.random() < TRUE_P[a] else 0.0
        counts[a] += 1
        Q[a] += (r - Q[a]) / counts[a]
        total_reg += BEST - TRUE_P[a]
        regret[t] = total_reg
    return regret


def run_ucb1(c=2.0):
    Q = np.zeros(K)
    counts = np.zeros(K)
    regret = np.zeros(T)
    total_reg = 0.0
    for t in range(T):
        if t < K:
            a = t
        else:
            ucb = Q + c * np.sqrt(np.log(t + 1) / counts)
            a = int(ucb.argmax())
        r = 1.0 if rng.random() < TRUE_P[a] else 0.0
        counts[a] += 1
        Q[a] += (r - Q[a]) / counts[a]
        total_reg += BEST - TRUE_P[a]
        regret[t] = total_reg
    return regret


def run_thompson():
    alpha = np.ones(K)
    beta = np.ones(K)
    regret = np.zeros(T)
    total_reg = 0.0
    for t in range(T):
        a = int((rng.beta(alpha, beta)).argmax())
        r = 1.0 if rng.random() < TRUE_P[a] else 0.0
        alpha[a] += r
        beta[a] += 1 - r
        total_reg += BEST - TRUE_P[a]
        regret[t] = total_reg
    return regret


def main():
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    ax.plot(run_epsilon(0.1), label="ε-greedy ε=0.1", color="#e6550d")
    ax.plot(run_epsilon(0.01), label="ε-greedy ε=0.01", color="#fdae6b")
    ax.plot(run_ucb1(), label="UCB1", color="#08519c")
    ax.plot(run_thompson(), label="Thompson", color="#31a354")
    ax.set_xlabel("round t")
    ax.set_ylabel("cumulative regret")
    ax.set_title("Bandit regret: log-time learners vs linear ε-greedy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
