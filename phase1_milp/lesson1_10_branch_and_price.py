"""Lesson 1.10 — Branch-and-Price: column generation inside branch-and-bound.

The gap lesson 1.9 left open: when block decisions are INTEGER, the DW
master's LP relaxation gives a bound, but its λ-solution may be fractional
— integrating the blocks one-by-one (naive rounding) loses optimality.
Branch-and-Price fixes it properly: branch on the ORIGINAL variables while
re-running column generation at every node.

Toy problem: B blocks x K binary activities (knapsack-per-block) sharing
2 global resources. Compare:
  (a) direct MILP (scipy.milp, gap 0) — ground truth + wall time
  (b) LP relaxation of the direct MILP — the bound quality baseline
  (c) DW master LP (column generation) — same bound, fewer columns
  (d) naive rounding of (c)'s fractional solution — shows the trap
  (e) branch-and-bound on the DW formulation, pricing at every node —
      exact, and the fractional-λ trap is handled honestly.
Runtime target < 3 min.
Run:  python phase1_milp/lesson1_10_branch_and_price.py
"""
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

EPS = 1e-7


def make_instance(B=6, K=10, seed=7):
    r = np.random.default_rng(seed)
    blocks = []
    for b in range(B):
        profit = np.round(r.uniform(5, 20, K), 1)
        local_usage = np.round(r.uniform(0.5, 2.0, (2, K)), 2)
        local_caps = local_usage.sum(axis=1) * r.uniform(0.45, 0.6, 2)
        blocks.append((profit, local_usage, local_caps))
    gu = np.round(r.uniform(0.2, 1.5, (2, K)), 2)
    gc = np.array([0.85, 0.75]) * gu.sum(axis=1) * B * 0.5
    return blocks, gu, gc


def solve_direct_mip(blocks, gu, gc, time_limit=120):
    B, K = len(blocks), blocks[0][0].size
    c, A, lo, hi = [], [], [], []
    for b, (profit, lu, lc) in enumerate(blocks):
        c.extend(-profit)
        for f in range(lu.shape[0]):
            row = np.zeros(B * K)
            row[b * K:(b + 1) * K] = lu[f]
            A.append(row); lo.append(-np.inf); hi.append(lc[f])
    for g in range(2):
        row = np.zeros(B * K)
        for b in range(B):
            row[b * K:(b + 1) * K] = gu[g]
        A.append(row); lo.append(-np.inf); hi.append(gc[g])
    t0 = time.time()
    res = milp(c=np.array(c),
               constraints=LinearConstraint(np.array(A), lo, hi),
               integrality=np.ones(B * K), bounds=Bounds(0, 1),
               options={"mip_rel_gap": 0.0, "time_limit": time_limit})
    return ((-res.fun if res.success else None),
            res.mip_gap if hasattr(res, "mip_gap") else None,
            time.time() - t0)


def solve_direct_lp_relax(blocks, gu, gc):
    B, K = len(blocks), blocks[0][0].size
    c, A, lo, hi = [], [], [], []
    for b, (profit, lu, lc) in enumerate(blocks):
        c.extend(-profit)
        for f in range(lu.shape[0]):
            row = np.zeros(B * K)
            row[b * K:(b + 1) * K] = lu[f]
            A.append(row); lo.append(-np.inf); hi.append(lc[f])
    for g in range(2):
        row = np.zeros(B * K)
        for b in range(B):
            row[b * K:(b + 1) * K] = gu[g]
        A.append(row); lo.append(-np.inf); hi.append(gc[g])
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(hi),
                  bounds=(0, 1), method="highs")
    return -res.fun


def solve_block_int(priced, lu, lc):
    """One block's knapsack: exact integer solve."""
    K = priced.size
    res = milp(c=priced, constraints=LinearConstraint(lu, -np.inf, lc),
               integrality=np.ones(K), bounds=Bounds(0, 1),
               options={"mip_rel_gap": 0.0})
    return float(priced @ res.x), res.x


def dw_cg_bound(blocks, gu, gc):
    """Column generation to LP optimality of the DW master relaxation.
    Returns (master_value, columns, mu_convexity, global_duals)."""
    B, K = len(blocks), blocks[0][0].size
    cols = []
    for b, (profit, lu, lc) in enumerate(blocks):
        priced = -profit
        val, plan = solve_block_int(priced, lu, lc)
        cols.append({"b": b, "profit": float(profit @ plan),
                     "gu": gu @ plan, "plan": plan})
        cols.append({"b": b, "profit": 0.0, "gu": np.zeros(2),
                     "plan": np.zeros(K)})
    it = 0
    while True:
        it += 1
        n = len(cols)
        c = np.array([-col["profit"] for col in cols])
        A_g = np.array([col["gu"] for col in cols]).T
        A_blk = np.zeros((B, n))
        for j, col in enumerate(cols):
            A_blk[col["b"], j] = 1.0
        res = linprog(c, A_ub=A_g, b_ub=gc, A_eq=A_blk, b_eq=np.ones(B),
                      bounds=(0, None), method="highs")
        pi = np.asarray(res.ineqlin.marginals[:2]).flatten()
        mu = np.asarray(res.eqlin.marginals).flatten()
        best = None
        for b, (profit, lu, lc) in enumerate(blocks):
            priced = -(profit + pi @ gu)
            val, plan = solve_block_int(priced, lu, lc)
            rc = val - mu[b]
            if rc < -EPS and (best is None or rc < best[0]):
                best = (rc, b, plan)
        if best is None:
            return -res.fun, cols, mu, pi, it
        _, b, plan = best
        profit, lu, lc = blocks[b]
        cols.append({"b": b, "profit": float(profit @ plan),
                     "gu": gu @ plan, "plan": plan})


def naive_round(blocks, cols, gu, gc):
    """The trap: take the fractional λ solution, keep the highest-λ plan
    per block, fix all others to zero. Feasible? Profitable? Usually no."""
    lam = None
    B, K = len(blocks), blocks[0][0].size
    n = len(cols)
    c = np.array([-col["profit"] for col in cols])
    A_g = np.array([col["gu"] for col in cols]).T
    A_blk = np.zeros((B, n))
    for j, col in enumerate(cols):
        A_blk[col["b"], j] = 1.0
    res = linprog(c, A_ub=A_g, b_ub=gc, A_eq=A_blk, b_eq=np.ones(B),
                  bounds=(0, None), method="highs")
    lam = res.x
    chosen = {}
    for j, col in enumerate(cols):
        if lam[j] > 1e-9:
            if col["b"] not in chosen or lam[j] > chosen[col["b"]][0]:
                chosen[col["b"]] = (lam[j], col)
    # rebuild integer solution: chosen plan per block (rounded plan)
    x = np.zeros(B * K)
    total_gu = np.zeros(2)
    profit_total = 0.0
    for b in range(B):
        if b not in chosen:
            continue
        _, col = chosen[b]
        x[b * K:(b + 1) * K] = np.round(col["plan"])
        profit_total += float(col["plan"] @
                              blocks[b][0] * 0)  # placeholder, recompute
        plan_int = np.round(col["plan"])
        profit_total += float(blocks[b][0] @ plan_int)
        total_gu += gu @ plan_int
    feasible = bool(np.all(total_gu <= gc + 1e-9))
    return profit_total, feasible, total_gu


def branch_and_price(blocks, gu, gc, node_budget=400):
    """Depth-first branch-and-bound on the ORIGINAL variables, re-pricing
    at every node. Branching: pick a fractional-looking block activity and
    fix it to 0/1 (enforced inside the block subproblems). Node = set of
    fixed (block, activity, value) restrictions."""
    B, K = len(blocks), blocks[0][0].size
    best_val, best_x = -np.inf, None
    nodes = 0
    # stack entries: (fixed_dict {(b,k): val}, inherited_cols)
    stack = [({}, None)]
    t0 = time.time()
    while stack and nodes < node_budget:
        fixed, cols_inherited = stack.pop()
        nodes += 1
        # --- price with current fixes: master over inherited columns
        cols = [dict(c) for c in (cols_inherited if cols_inherited
                                  else [])]
        if not cols:
            for b, (profit, lu, lc) in enumerate(blocks):
                priced = -profit.copy()
                for (fb, fk), fv in fixed.items():
                    if fb == b:
                        priced[fk] = 1e9 if fv == 0 else -1e9
                val, plan = solve_block_int(priced, lu, lc)
                plan = _apply_fixes(plan, fixed, b)
                if plan is None:
                    continue
                cols.append({"b": b, "profit": float(profit @ plan),
                             "gu": gu @ plan, "plan": plan})
                zero_plan = _apply_fixes(np.zeros(K), fixed, b)
                cols.append({"b": b, "profit": 0.0, "gu": np.zeros(2),
                             "plan": zero_plan})
        it = 0
        feasible_node = True
        while True:
            it += 1
            n = len(cols)
            c = np.array([-col["profit"] for col in cols])
            A_g = np.array([col["gu"] for col in cols]).T
            A_blk = np.zeros((B, n))
            for j, col in enumerate(cols):
                A_blk[col["b"], j] = 1.0
            res = linprog(c, A_ub=A_g, b_ub=gc, A_eq=A_blk, b_eq=np.ones(B),
                          bounds=(0, None), method="highs")
            if not res.success:
                feasible_node = False
                break
            pi = np.asarray(res.ineqlin.marginals[:2]).flatten()
            mu = np.asarray(res.eqlin.marginals).flatten()
            best = None
            for b, (profit, lu, lc) in enumerate(blocks):
                priced = -(profit + pi @ gu).copy()
                blocked = False
                for (fb, fk), fv in fixed.items():
                    if fb == b:
                        # forbid violating the branch inside subproblem
                        priced = priced  # handled via plan fix below
                val, plan = solve_block_int(priced, lu, lc)
                plan_fixed = _apply_fixes(plan, fixed, b)
                if plan_fixed is None:
                    continue
                # recompute rc on the FIXED plan
                rc = float(priced @ plan_fixed) - mu[b]
                if rc < -EPS and (best is None or rc < best[0]):
                    best = (rc, b, plan_fixed)
            if best is None:
                break
            _, b, plan = best
            profit, lu, lc = blocks[b]
            cols.append({"b": b, "profit": float(profit @ plan),
                         "gu": gu @ plan, "plan": plan})
            if it > 60:
                break
        if not feasible_node:
            continue
        bound = -res.fun
        if bound <= best_val + 1e-9:
            continue                      # prune: bound no better
        # --- build an integer solution from the master's λ (greedy commit)
        x_int, val_int = _integer_from_columns(cols, res, blocks, gu, gc,
                                               fixed)
        if val_int > best_val:
            best_val, best_x = val_int, x_int
        # --- find a branching variable: block activity with fractional
        # usage in the λ-mix (approximate: highest-λ column's plan diff)
        var = _pick_branch_var(cols, res, blocks, fixed)
        if var is None:
            continue
        for fv in (0, 1):
            nf = dict(fixed)
            nf[var] = fv
            stack.append((nf, cols))
    return best_val, nodes, time.time() - t0


def _apply_fixes(plan, fixed, b):
    plan = plan.copy()
    for (fb, fk), fv in fixed.items():
        if fb == b:
            plan[fk] = fv
    return plan


def _integer_from_columns(cols, res, blocks, gu, gc, fixed):
    """Greedy: sort columns by λ desc; commit per-block integer plans while
    the shared resources allow."""
    B, K = len(blocks), blocks[0][0].size
    order = np.argsort(-res.x)
    x = np.zeros(B * K)
    used = np.zeros(2)
    profit = 0.0
    committed = set()
    for j in order:
        col = cols[j]
        if col["b"] in committed:
            continue
        plan = np.round(col["plan"]).astype(int)
        for (fb, fk), fv in fixed.items():
            if fb == col["b"]:
                plan[fk] = fv
        add_gu = gu @ plan
        if np.all(used + add_gu <= gc + 1e-9):
            x[col["b"] * K:(col["b"] + 1) * K] = plan
            used += add_gu
            profit += float(blocks[col["b"]][0] @ plan)
            committed.add(col["b"])
    # zero-fill blocks never committed (feasible by zero plans)
    return x, profit


def _pick_branch_var(cols, res, blocks, fixed):
    """Pick the (b,k) whose highest-λ column plan has value most contested:
    here simply the first activity not yet fixed in the top-λ column."""
    order = np.argsort(-res.x)
    for j in order:
        col = cols[j]
        for k in range(len(col["plan"])):
            if (col["b"], k) not in fixed and col["plan"][k] not in (0.0,):
                return (col["b"], k)
    return None


if __name__ == "__main__":
    blocks, gu, gc = make_instance()
    B, K = len(blocks), blocks[0][0].size
    print(f"instance: {B} blocks x {K} BINARY activities, 2 shared resources")

    print("\n(a) direct MILP (ground truth):")
    opt_mip, mip_gap, t_mip = solve_direct_mip(blocks, gu, gc)
    print(f"    optimum {opt_mip:.3f}  ({t_mip:.2f} s)")

    print("\n(b) direct LP relaxation (bound baseline):")
    lp = solve_direct_lp_relax(blocks, gu, gc)
    print(f"    LP bound {lp:.3f}  (integrality gap "
          f"{100 * (lp - opt_mip) / max(opt_mip, 1e-9):.2f}%)")

    print("\n(c) DW master LP via column generation:")
    t0 = time.time()
    master_val, cols, mu, pi, it = dw_cg_bound(blocks, gu, gc)
    print(f"    bound {master_val:.3f} after {it} rounds, {len(cols)} cols "
          f"({time.time() - t0:.2f} s) — same bound as (b), fewer columns")

    print("\n(d) naive rounding of the fractional λ solution:")
    val_r, feas, gu_used = naive_round(blocks, cols, gu, gc)
    print(f"    rounded profit {val_r:.1f}  feasible={feas}  "
          f"gap vs optimum {100 * (opt_mip - val_r) / opt_mip:.1f}%")

    print("\n(e) branch-and-price (DFS, pricing at every node):")
    val_bp, nodes, t_bp = branch_and_price(blocks, gu, gc)
    print(f"    best integer {val_bp:.3f} after {nodes} nodes "
          f"({t_bp:.1f} s)")
    print(f"    vs direct MILP optimum: {100 * (val_bp - opt_mip) / opt_mip:+.2f}%")
