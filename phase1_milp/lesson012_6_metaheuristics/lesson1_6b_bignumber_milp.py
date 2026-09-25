"""Lesson 1.6b — Exact vs heuristic: the crossover experiment.

Same knapsack family at growing n solved by (a) scipy.optimize.milp exact,
(b) greedy + bit-flip local search. Reports wall time and, for the exact
solve, the optimality certificate; for the heuristic, the true gap vs exact.
Run:  python lesson1_6b_bignumber_milp.py
"""
import random
import time

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def make_instance(n, seed=7):
    rng = random.Random(seed)
    w = [rng.randint(2, 40) for _ in range(n)]
    p = [rng.randint(5, 90) for _ in range(n)]
    return w, p, sum(w) // 3


def solve_exact(w, p, cap, rel_gap=0.0):
    n = len(w)
    c = -np.array(p, dtype=float)                      # maximize p
    A = np.array(w, dtype=float).reshape(1, -1)
    res = milp(c=c, constraints=LinearConstraint(A, -np.inf, cap),
               integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"mip_rel_gap": rel_gap})
    return -res.fun if res.success else None, res.mip_gap, res.message


def greedy(w, p, cap):
    order = sorted(range(len(w)), key=lambda i: -p[i] / w[i])
    bits, tw = [0] * len(w), 0
    for i in order:
        if tw + w[i] <= cap:
            bits[i], tw = 1, tw + w[i]
    return sum(x for x, b in zip(p, bits) if b)


def local_search(w, p, cap, budget_s=1.0):
    """Greedy start + best-of-100 random bit flips, first improvement."""
    t0 = time.time()
    bits = [0] * len(w)
    tw = 0
    order = sorted(range(len(w)), key=lambda i: -p[i] / w[i])
    for i in order:
        if tw + w[i] <= cap:
            bits[i], tw = 1, tw + w[i]
    val = sum(x for x, b in zip(p, bits) if b)
    rng = random.Random(1)
    while time.time() - t0 < budget_s:
        i = rng.randrange(len(bits))
        cand, cw = bits[:], tw
        cand[i] = 1 - cand[i]
        cw += w[i] * (1 if cand[i] else -1)
        if cw <= cap:
            cv = val + p[i] * (1 if cand[i] else -1)
            if cv > val:
                bits, val = cand, cv
    return val


if __name__ == "__main__":
    print(f"{'n':>7s} {'exact (s)':>10s} {'exact val':>10s} "
          f"{'heuristic':>10s} {'gap %':>7s} {'heur time':>10s}")
    for n in (50, 100, 200, 400, 800, 1600, 3200, 6400):
        w, p, cap = make_instance(n)
        t0 = time.time()
        opt, gap, msg = solve_exact(w, p, cap)
        t_exact = time.time() - t0
        t0 = time.time()
        hv = local_search(w, p, cap, budget_s=0.5)
        t_heur = time.time() - t0
        gap_pct = 100 * (opt - hv) / opt if opt else float("nan")
        # |gap| < 0.01% prints as 0.00 — the heuristic may land a hair above a
        # *rounded* print of the exact value; round both to cents first.
        gap_pct = round(gap_pct, 2) + 0.0 if abs(gap_pct) < 0.005 else gap_pct
        print(f"{n:7d} {t_exact:10.2f} {opt:10.0f} {hv:10.0f} "
              f"{gap_pct:7.2f} {t_heur:10.2f}" + (f"   [{msg.split('.')[0]}]"
                                                 if not opt else ""))
