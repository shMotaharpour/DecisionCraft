"""Lesson 1.9 — Dantzig-Wolfe Decomposition & Column Generation.

Motivating structure: B independent production units (blocks), each with
its OWN local constraints (machine hours per unit), sharing a few GLOBAL
resources (budget, logistics). Direct LP: one big block-diagonal system.
DW: each block's feasible plans = "columns"; a master LP picks the mix
subject to the shared resources; subproblems price out new columns with
the master's duals until none improves (proven: exact LP method, finite).

(a) ground truth: solve the full LP directly (scipy.linprog)
(b) DW + column generation from ONE trivial column per block
(c) report iterations, columns generated, optimality match, and the
    dual-pricing story (reduced cost = block profit − duals · global usage)
Run:  python phase1_milp/lesson1_9_dantzig_wolfe.py
"""
import time

import numpy as np
from scipy.optimize import linprog

EPS = 1e-7


def make_instance(B=6, K=8, F=4, seed=7):
    """B blocks; each block has K activities and F local constraints.
    Two shared resources couple the blocks."""
    r = np.random.default_rng(seed)
    blocks = []
    for b in range(B):
        profit = r.uniform(5, 20, K)
        local_usage = r.uniform(0.5, 2.0, (F, K))
        local_caps = local_usage.sum(axis=1) * r.uniform(0.35, 0.6, F)
        blocks.append((profit, local_usage, local_caps))
    gu = np.stack([r.uniform(0.2, 1.5, K) for _ in range(2)])
    gc = np.array([0.9, 0.8]) * gu.sum(axis=1) * B * 0.55
    return blocks, gu, gc


def solve_direct(blocks, gu, gc):
    """Ground truth: the whole block-diagonal LP in one call."""
    B, K = len(blocks), blocks[0][0].size
    c, A_ub, b_ub = [], [], []
    for b, (profit, lu, lc) in enumerate(blocks):
        c.extend(-profit)
        for f in range(lu.shape[0]):
            row = np.zeros(B * K)
            row[b * K:(b + 1) * K] = lu[f]
            A_ub.append(row)
            b_ub.append(lc[f])
    for g in range(2):
        row = np.zeros(B * K)
        for b in range(B):
            row[b * K:(b + 1) * K] = gu[g]
        A_ub.append(row)
        b_ub.append(gc[g])
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=(0, None), method="highs")
    return -res.fun, res


def dw_solve(blocks, gu, gc, verbose=True):
    """Dantzig-Wolfe with column generation. Master:
        min  Σ  (−profit_j) λ_j
        s.t. Σ  gu_j λ_j ≤ gc          (2 shared resources — the duals!)
             Σ_{j ∈ block b} λ_j = 1   (convexity per block)
             λ ≥ 0
    Initial columns: (i) each block's max-profit plan under local
    constraints ONLY, and (ii) each block's zero-usage plan (all zeros —
    always locally feasible since caps > 0). The zero plan guarantees the
    initial master is feasible (standard big-M/phase-1 alternative).
    """
    B = len(blocks)
    cols = []
    for b, (profit, lu, lc) in enumerate(blocks):
        res = linprog(-profit, A_ub=lu, b_ub=lc, bounds=(0, None),
                      method="highs")
        cols.append({"b": b, "profit": float(profit @ res.x),
                     "gu": gu @ res.x, "plan": res.x})
        # zero plan: buys feasibility for the initial master (profit 0)
        cols.append({"b": b, "profit": 0.0,
                     "gu": np.zeros(gu.shape[0]),
                     "plan": np.zeros(profit.size)})

    t0 = time.time()
    it = 0
    for it in range(1, 201):
        # ---- master LP over existing columns
        n = len(cols)
        c = np.array([-col["profit"] for col in cols])       # max profit
        A_g = np.array([col["gu"] for col in cols]).T        # 2 x n
        A_blk = np.zeros((B, n))
        for j, col in enumerate(cols):
            A_blk[col["b"], j] = 1.0
        res = linprog(c,
                      A_ub=A_g, b_ub=gc,
                      A_eq=A_blk, b_eq=np.ones(B),
                      bounds=(0, None), method="highs")
        # scipy marginals for a min-problem: pi <= 0 for <= rows.
        # Reduced cost of a candidate plan x in block b:
        #   rc(x) = -(profit + pi·gu)·x − mu_b     (min convention)
        # so the subproblem MINIMIZES -(profit + pi·gu)·x, i.e. maximizes
        # the dual-priced profit (profit + pi·gu)·x. Sign verified live.
        pi = np.asarray(res.ineqlin.marginals[:2]).flatten()
        mu = np.asarray(res.eqlin.marginals).flatten()
        if verbose and (it <= 3 or it in (10, 50, 100)):
            print(f"   it {it:3d}: obj {-res.fun:9.3f}  pi "
                  f"{np.round(pi, 3)}  cols {n}")

        # ---- pricing: each block solves its own LP with the duals
        best = None
        for b, (profit, lu, lc) in enumerate(blocks):
            c_sub = -(profit + pi @ gu)
            sub = linprog(c_sub, A_ub=lu, b_ub=lc, bounds=(0, None),
                          method="highs")
            rc = float(c_sub @ sub.x) - mu[b]
            if rc < -EPS and (best is None or rc < best[0]):
                best = (rc, b, sub.x)
        if best is None:
            break                     # no improving column anywhere: done
        _, b, plan = best
        profit, lu, lc = blocks[b]
        cols.append({"b": b, "profit": float(profit @ plan),
                     "gu": gu @ plan, "plan": plan})
    return -res.fun, it, len(cols), time.time() - t0, res


if __name__ == "__main__":
    blocks, gu, gc = make_instance()
    print(f"instance: {len(blocks)} blocks x {blocks[0][0].size} activities, "
          f"2 shared resources")

    t0 = time.time()
    opt_direct, _ = solve_direct(blocks, gu, gc)
    t_direct = time.time() - t0
    print(f"\n(a) direct LP   : profit {opt_direct:.4f}  ({t_direct*1000:.0f} ms)")

    print("\n(b) Dantzig-Wolfe + column generation:")
    opt_dw, iters, ncols, t_dw, _ = dw_solve(blocks, gu, gc)
    print(f"    converged in {iters} pricing rounds, {ncols} columns, "
          f"{t_dw*1000:.0f} ms")
    print(f"    profit {opt_dw:.4f}")

    print(f"\n(c) |direct − DW| = {abs(opt_direct - opt_dw):.2e} "
          "(proven: same LP, different path)")
    print("    the loop that closed: master duals = shadow prices of the")
    print("    shared resources (lesson 1.4) -> each block best-responds")
    print("    locally (its own LP) -> the master re-prices. This is the")
    print("    formal core of the lesson-4.3 hybrid architecture.")
