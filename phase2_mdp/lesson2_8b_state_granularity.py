"""Lesson 2.8b — State granularity: the resolution dial on the STATE axis.

Same inventory MDP (Poisson demand, (s,S)-friendly). Three state
resolutions:
  exact     — stock 0..20, 21 states
  bin(3)    — stock lumped into 7 bins {0-2, 3-5, ...}
  bin(10)   — stock lumped into 3 bins
Coarse models solve faster and generalize, but the policy they return
must be LIFTED back to the exact state space to deploy. Measures:
solve time, policy's true performance when executed in the exact env
(honest evaluation, not the coarse model's self-estimate).

Run: python phase2_mdp/lesson2_8b_state_granularity.py   (~30 s)
"""
import time

import numpy as np
from scipy.stats import poisson

GAMMA = 0.95
MAX_S = 20                 # true stock range 0..20
N_EXACT = MAX_S + 1
PRICE, COST, HOLD = 10.0, 3.0, 0.4
LAM = 3.0
N_EVAL = 20000


def exact_pmf():
    pmf = poisson.pmf(np.arange(MAX_S + 1), LAM)
    pmf[-1] += 1 - pmf.sum()
    return pmf


def solve(stock_states, stock_action_max_per_state):
    """VI over an abstracted stock set. stock_states: list of abstract
    stock levels; transition uses the TRUE dynamics evaluated at that
    level (a pseudo-discretization: coarse grid on the same physics)."""
    NS = len(stock_states)
    NA = max(stock_action_max_per_state) + 1
    pmf = exact_pmf()
    P = np.zeros((NS, NA, NS))
    R = np.full((NS, NA), -1e9)
    for i, s in enumerate(stock_states):
        for a in range(stock_action_max_per_state[i] + 1):
            R[i, a] = sum(
                pr * (PRICE * min(s + a, d) - HOLD * max(s + a - d, 0))
                for d, pr in enumerate(pmf)) - COST * a
            for d, pr in enumerate(pmf):
                s2 = max(0, s + a - d)
                j = int(np.argmin(np.abs(stock_states - s2)))  # project
                P[i, a, j] += pr
    pol = np.zeros(NS, dtype=int)
    V = np.zeros(NS)
    for _ in range(300):
        Q = R + GAMMA * np.einsum("iap,p->ia", P, V)
        Vn = Q.max(axis=1)
        new = Q.argmax(axis=1)
        if np.allclose(new, pol):
            pol = new
            V = Vn
            break
        pol, V = new, Vn
    return V, pol, stock_states


def lift_and_evaluate(pol, states):
    """Deploy the coarse policy in the TRUE env: order = the abstract
    level's action; evaluate honestly on exact dynamics."""
    pmf = exact_pmf()
    rng = np.random.default_rng(4)
    total = 0.0
    for _ in range(N_EVAL):
        s = int(rng.integers(0, N_EXACT))
        g, G = 1.0, 0.0
        for _t in range(60):
            i = int(np.argmin(np.abs(states - s)))
            a = int(pol[i])
            a = min(a, MAX_S - s)
            d = int(rng.choice(MAX_S + 1, p=pmf))
            sold = min(s + a, d)
            r = PRICE * sold - HOLD * max(s + a - d, 0) - COST * a
            G += g * r
            g *= GAMMA
            s = max(0, s + a - d)
        total += G
    return total / N_EVAL


def main():
    specs = [
        ("exact    (21 levels)", np.arange(N_EXACT), [MAX_S - s for s in range(N_EXACT)]),
        ("bin(3)   ( 7 levels)", np.arange(0, MAX_S + 1, 3),
         [MAX_S - s for s in np.arange(0, MAX_S + 1, 3)]),
        ("bin(10)  ( 3 levels)", np.array([0, 10, 20]),
         [MAX_S - s for s in [0, 10, 20]]),
    ]
    print(f"State granularity — inventory, gamma={GAMMA}, "
          f"evaluated over {N_EVAL} honest rollouts each\n")
    rows = []
    for name, states, amax in specs:
        t0 = time.perf_counter()
        V, pol, states = solve(np.asarray(states), amax)
        dt = time.perf_counter() - t0
        perf = lift_and_evaluate(pol, states)
        rows.append((name, len(states), dt, perf))
        print(f"{name}  |S|={len(states):3d}  solve {dt*1000:6.1f} ms  "
              f"true perf={perf:8.2f}")
    best = max(r[3] for r in rows)
    print("\nread:")
    print(" - the coarse model's OWN value function is optimistic garbage")
    print("   (projection error); only the lifted rollout tells the truth.")
    print(" - bin(3) loses a few percent; bin(10) is visibly degraded —")
    print("   the (s,S) structure needs enough resolution to place the")
    print("   reorder point.")
    print(" - rule of thumb: bin where the value function is FLAT, keep")
    print("   resolution where it bends (the reorder region).")


if __name__ == "__main__":
    main()
