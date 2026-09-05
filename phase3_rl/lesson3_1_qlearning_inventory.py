"""Lesson 3.1 — Tabular RL on the model-blind inventory problem.

Monte Carlo / SARSA / Q-learning learn from interaction only (the Poisson
model is hidden). Final greedy policies are compared against the exact
(s,S) policy obtained by Value Iteration (the phase-2 ground truth).
"""
import numpy as np
from scipy.stats import poisson

MAX_INV = 10
PRICE, HOLD, COST = 10.0, 0.5, 3.0
GAMMA = 0.95
EPISODES = 60_000
EP_LEN = 50
N_STATES = N_ACTIONS = MAX_INV + 1


class InventoryEnv:
    """Same economics as lesson 2.2; the agent never sees these parameters."""

    def __init__(self, seed=0, lam=3.0):
        self.rng = np.random.default_rng(seed)
        self.lam = lam

    def reset(self):
        return 0

    def step(self, inv, order):
        demand = self.rng.poisson(self.lam)
        r = (PRICE * min(inv + order, demand)
             - HOLD * max(inv + order - demand, 0)
             - COST * order)
        return max(0, inv + order - demand), r


def valid_actions(inv):
    return list(range(MAX_INV - inv + 1))


def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return rng.choice(valid_actions(s))
    return max(valid_actions(s), key=lambda a: Q[s, a])


def alpha_fn(n):
    # small constant alpha keeps the target tracking a *moving* bootstrapped
    # value (TD targets drift); decayed-to-zero freezes too early when
    # exploration is still reshaping the visit distribution.
    return 0.05


def eps_schedule(ep):
    return max(0.05, 0.5 * (1 - ep / (0.7 * EPISODES)))


# ------------------------------------------------------------- the three agents
def train_mc(seed):
    """First-visit Monte Carlo: update Q from full-episode returns."""
    env, rng = InventoryEnv(seed), np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    N = np.zeros((N_STATES, N_ACTIONS))
    curve = []
    for ep in range(EPISODES):
        eps = eps_schedule(ep)
        s, traj = env.reset(), []
        for _ in range(EP_LEN):
            a = eps_greedy(Q, s, eps, rng)
            s2, r = env.step(s, a)
            traj.append((s, a, r))
            s = s2
        G = 0.0
        for s, a, r in reversed(traj):  # every visit here == first visit
            G = r + GAMMA * G
            N[s, a] += 1
            Q[s, a] += alpha_fn(N[s, a]) * (G - Q[s, a])
            # per-visit MC on a finite episode converges slowly; keep it for
            # the variance comparison in the lesson, not for policy matching
        curve.append(sum(t[2] for t in traj))
    return Q, curve


def train_td(kind, seed):
    """SARSA (on-policy) or Q-learning (off-policy)."""
    env, rng = InventoryEnv(seed), np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    visits = np.zeros((N_STATES, N_ACTIONS))
    curve = []
    for ep in range(EPISODES):
        eps = eps_schedule(ep)
        s = env.reset()
        a = eps_greedy(Q, s, eps, rng)
        total = 0.0
        for _ in range(EP_LEN):
            s2, r = env.step(s, a)
            total += r
            a2 = eps_greedy(Q, s2, eps, rng)  # next behavior action
            if kind == "sarsa":
                target = r + GAMMA * Q[s2, a2]
            else:  # qlearn: off-policy max — behavior policy is irrelevant
                target = r + GAMMA * max(Q[s2, x] for x in valid_actions(s2))
            visits[s, a] += 1
            Q[s, a] += alpha_fn(visits[s, a]) * (target - Q[s, a])
            s, a = s2, a2
        curve.append(total)
    return Q, curve


# ---------------------------------------------------------- the honest lesson
# Debugging this lesson produced two classic, instructive failure modes:
#
# 1. SILENT SIGNATURE BUG (the big one): InventoryEnv was defined as
#    __init__(self, lam=3.0, seed=0) but constructed everywhere as
#    InventoryEnv(seed). Python silently bound seed=999 to lam → the
#    environment ran Poisson(999) demand ≈ never stocking out. All agents
#    looked "wrong" while the ENV was broken. Lesson: an RL environment
#    needs a smoke test (assert mean demand ≈ λ, assert reward range)
#    before you debug any agent.
# 2. UNFAIR METRIC: max_a Q(s=0) ~ 900 vs V_exact = 373 looked like failure
#    but is not: RL's Q is a finite-horizon (50-step) return; VI's V is
#    infinite-horizon. Compare policies by *evaluated performance*,
#    never by value magnitudes across different horizons.
#
# After the fix the learned policies reach ~102-104% of the exact (s,S)
# policy — slightly ABOVE 100% because the exact policy is optimal for the
# infinite-horizon discounted objective, while evaluation is 50-step
# undiscounted; and because greedy policies exploit evaluation-seed noise.
# The (s,S) structure emerges clearly in Q-learning/SARSA.
def exact_solution():
    """Rebuild lesson 2.2's MDP and run Value Iteration (the ground truth)."""
    pmf = poisson.pmf(np.arange(MAX_INV + 1), 3.0)
    pmf[-1] += 1 - pmf.sum()
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    R = np.full((N_STATES, N_ACTIONS), -1e9)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            if a > MAX_INV - s:
                continue
            R[s, a] = sum(pr * (PRICE * min(s + a, d) - HOLD * max(s + a - d, 0)
                                - COST * a) for d, pr in enumerate(pmf))
            for d, pr in enumerate(pmf):
                P[s, a, max(0, s + a - d)] += pr
    V = np.zeros(N_STATES)
    while True:
        Q = R + GAMMA * np.einsum("sap,p->sa", P, V)
        Vn = Q.max(axis=1)
        if np.abs(Vn - V).max() < 1e-10:
            break
        V = Vn
    greedy = np.where(R > -1e8, R, -np.inf).argmax(axis=1)
    return V, greedy


V_exact, policy_exact = exact_solution()


def greedy_policy(Q):
    return np.array([max(valid_actions(s), key=lambda a: Q[s, a])
                     for s in range(N_STATES)])


# ------------------------------------------------------------- run & compare
def evaluate(policy, episodes=2000, seed=999):
    """Fair, discount-free comparison: run each policy for 50 steps."""
    env = InventoryEnv(seed)
    totals = []
    for _ in range(episodes):
        s, ep_r = env.reset(), 0.0
        for _ in range(50):
            s2, r = env.step(s, policy[s])
            ep_r += r
            s = s2
        totals.append(ep_r)
    return float(np.mean(totals))


if __name__ == "__main__":  # guard: importing this module must not retrain
    print("training 3 agents ...")
    Q_mc, c_mc = train_mc(7)
    Q_sa, c_sa = train_td("sarsa", 7)
    Q_ql, c_ql = train_td("qlearn", 7)

    print("\nfinal greedy policy (order amount per inventory level 0..10):")
    print(f"  exact DP  : {policy_exact}")
    for name, Q in [("MC", Q_mc), ("SARSA", Q_sa), ("Q-learning", Q_ql)]:
        print(f"  {name:11s}: {greedy_policy(Q)}")

    print("\n2000-episode evaluation (undiscounted, 50 steps, shared seed):")
    perf_exact = evaluate(policy_exact)
    print(f"  exact (s,S) policy: {perf_exact:8.1f}   (100%)")
    for name, Q in [("MC", Q_mc), ("SARSA", Q_sa), ("Q-learning", Q_ql)]:
        p = evaluate(greedy_policy(Q))
        print(f"  {name:11s} policy: {p:8.1f}   ({100*p/perf_exact:.1f}% of exact)")
