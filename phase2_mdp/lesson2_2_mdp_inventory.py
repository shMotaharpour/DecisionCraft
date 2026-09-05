"""Lesson 2.2 — MDP: inventory problem solved exactly with Value Iteration
and Policy Iteration (NumPy only).

State s = inventory (0..10), action a = order (0..10-s), demand ~ Poisson(3).
Reward: p*min(inv, D) - h*max(inv-D, 0) - c*a, discounted by gamma.
"""
import itertools

import numpy as np
from scipy.stats import poisson

MAX_INV = 10
LAMBDA = 3.0
PRICE, HOLD, COST = 10.0, 0.5, 3.0
GAMMA = 0.95

S = np.arange(MAX_INV + 1)  # states: inventory levels

# Demand pmf, truncated at MAX_INV (remaining mass lumped on the last value)
pmf = poisson.pmf(np.arange(0, MAX_INV + 1), LAMBDA)
pmf[-1] += 1.0 - pmf.sum()  # D >= MAX_INV behaves as D == MAX_INV


def next_inv(inv: int, order: int, demand: int) -> int:
    return max(0, inv + order - demand)


def reward(inv: int, order: int) -> float:
    """Expected immediate reward over demand D (expectation over pmf)."""
    total = 0.0
    for d, pr in enumerate(pmf):
        sold = min(inv + order, d)
        leftover = max(inv + order - d, 0)
        total += pr * (PRICE * sold - HOLD * leftover - COST * order)
    return total


# -------------------------------------------------- build P(s'|s,a), R(s,a)
nA = MAX_INV + 1  # actions 0..MAX_INV (invalid ones masked below)
P = np.zeros((MAX_INV + 1, nA, MAX_INV + 1))  # [s, a, s']
R = np.full((MAX_INV + 1, nA), -1e9)          # invalid actions = -inf

for s in S:
    for a in range(nA):
        if a > MAX_INV - s:
            continue  # cannot overfill warehouse
        R[s, a] = reward(s, a)
        for d, pr in enumerate(pmf):
            P[s, a, next_inv(s, a, d)] += pr

# -------------------------------------------------- Value Iteration
def value_iteration(tol=1e-9, max_iter=10_000):
    V = np.zeros(len(S))
    for it in range(max_iter):
        # T(V)(s) = max_a R[s,a] + gamma * P[s,a,:] @ V   (masked by -1e9 R)
        Q = R + GAMMA * np.einsum("sap,p->sa", P, V)
        V_new = Q.max(axis=1)
        if np.abs(V_new - V).max() < tol:
            return V_new, Q, it
        V = V_new
    return V, Q, max_iter


# -------------------------------------------------- Policy Iteration
def policy_iteration(max_iter=100):
    policy = np.zeros(len(S), dtype=int)  # start: never order
    for it in range(max_iter):
        # policy evaluation: solve (I - gamma*P_pi) V = R_pi  (linear system!)
        P_pi = P[np.arange(len(S)), policy]        # [s, s']
        R_pi = R[np.arange(len(S)), policy]
        V = np.linalg.solve(np.eye(len(S)) - GAMMA * P_pi, R_pi)
        # policy improvement
        Q = R + GAMMA * np.einsum("sap,p->sa", P, V)
        new_policy = Q.argmax(axis=1)
        if np.array_equal(new_policy, policy):
            return V, policy, it
        policy = new_policy
    raise RuntimeError("PI did not converge")


V_vi, Q, vi_iters = value_iteration()
V_pi, pi, pi_iters = policy_iteration()

print(f"Value Iteration: converged in {vi_iters} sweeps")
print(f"Policy Iteration: converged in {pi_iters} iterations")
print(f"|V_vi - V_pi| max = {np.abs(V_vi - V_pi).max():.2e}  (same V*)")

print("\nOptimal policy (s -> order up to s+a):")
for s in S:
    print(f"  inv={s:2d} -> order {pi[s]:2d}  (target level {s + pi[s]:2d})  V={V_pi[s]:8.2f}")

# sanity: (s,S) structure — target level should be constant for low inventory
targets = [s + pi[s] for s in S if pi[s] > 0]
print(f"\n(s,S) rule check: order-up-to levels used: {sorted(set(targets))}")
print(f"Reorder point s*: smallest s with order > 0 = {min(s for s in S if pi[s] > 0)}")
