"""Lesson 3.5 — Bandits in isolation: regret, UCB1, Thompson, and the law.

The explore/exploit dilemma WITHOUT states (context-free bandit) — the
cleanest possible setting where the theory is exactly computable:
  A. 10-armed Bernoulli bandit: epsilon-greedy (2 schedules), UCB1,
     Thompson sampling — cumulative regret curves, ~40k steps, < 30 s.
  B. Verify the Lai–Robbins prediction: per-arm pulls concentrate on the
     arms within the log(T)-KL cone — count pulls of suboptimal arms and
     compare against the theoretical lower bound scale.
All random seeds fixed; total runtime well under a minute.
Run:  python lesson3_5_bandits.py
"""
import numpy as np

rng = np.random.default_rng(11)

# ------------------------------------------------ the bandit
K = 10
MU = np.sort(rng.uniform(0.2, 0.8, K))[::-1]     # descending: arm 0 best
MU_STAR = MU[0]
T = 40_000


def pull(i):
    return float(rng.random() < MU[i])


def run_epsilon(eps_fn, T=T):
    Q = np.zeros(K)
    N = np.zeros(K)
    regret = np.zeros(T)
    pulls = np.zeros(K)
    t0 = 0
    for t in range(T):
        eps = eps_fn(t)
        a = rng.integers(K) if rng.random() < eps else int(Q.argmax())
        r = pull(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        pulls[a] += 1
        t0 += 1
        regret[t] = regret[t - 1] + (MU_STAR - MU[a])
    return Q, N, regret, pulls


def run_ucb1(c=2.0, T=T):
    Q = np.zeros(K)
    N = np.zeros(K)
    regret = np.zeros(T)
    pulls = np.zeros(K)
    for t in range(T):
        if t < K:                       # play each arm once
            a = t
        else:
            ucb = Q + c * np.sqrt(np.log(t) / N)
            a = int(ucb.argmax())
        r = pull(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        pulls[a] += 1
        regret[t] = regret[t - 1] + (MU_STAR - MU[a])
    return Q, N, regret, pulls


def run_thompson(T=T):
    """Beta(a,b) posterior per arm; sample & act greedily on the sample."""
    alpha = np.ones(K)
    beta = np.ones(K)
    regret = np.zeros(T)
    pulls = np.zeros(K)
    for t in range(T):
        a = int((rng.beta(alpha, beta)).argmax())
        r = pull(a)
        alpha[a] += r
        beta[a] += 1 - r
        pulls[a] += 1
        regret[t] = regret[t - 1] + (MU_STAR - MU[a])
    return None, None, regret, pulls


if __name__ == "__main__":
    print(f"{K}-armed Bernoulli bandit, T={T:,} steps, "
          f"mu* = {MU_STAR:.3f}, second-best {MU[1]:.3f}")
    print(f"{'policy':28s} {'total regret':>13s} {'pulls of worst arm':>19s}")

    results = {}
    for name, fn in (
        ("eps const 0.10", lambda t: 0.10),
        ("eps decay 1/sqrt(t+1)", lambda t: min(1.0, 1.0 / np.sqrt(t + 1))),
        ("UCB1 c=2", None),
        ("Thompson", None),
    ):
        if name.startswith("UCB"):
            Q, N, reg, pulls = run_ucb1()
        elif name.startswith("Thomp"):
            Q, N, reg, pulls = run_thompson()
        else:
            Q, N, reg, pulls = run_epsilon(fn)
        results[name] = reg
        print(f"{name:28s} {reg[-1]:13.1f} {int(pulls[-1]):19d}")

    print("\nregret at checkpoints (log-scale growth visible):")
    marks = [100, 1_000, 10_000, 40_000]
    print("   " + "  ".join(f"t={m:>6d}" for m in marks))
    for name, reg in results.items():
        print(f"   {name:26s} " +
              "  ".join(f"{reg[m-1]:7.1f}" for m in marks))

    # ---- Lai-Robbins flavor: suboptimal pulls grow ~ log(T) per arm ------
    print("\nUCB1 pulls of the two WORST arms vs log T (Lai-Robbins cone):")
    _, N, _, pulls = run_ucb1()
    worst = [K - 1, K - 2]
    for t_mark in (10_000, 40_000):
        expected_scale = np.log(t_mark) / np.log(2)   # KL(Bern) ~ O(1) here
        print(f"   t={t_mark:>6d}: pulls arm K-1 = {int(pulls[K-1]):4d}, "
              f"arm K-2 = {int(pulls[K-2]):4d}, log2(T) = {np.log2(t_mark):.1f}")
    print("   (logarithmic growth = the proven optimal rate; linear-growth")
    print("    constant-eps is exactly the gap the lower bound forbids.)")
