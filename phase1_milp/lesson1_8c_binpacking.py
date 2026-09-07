"""Lesson 1.8c — Bin packing / cutting stock: first-fit heuristics + CP-SAT.

  A. Three classic heuristics from scratch on one seeded instance:
     Next Fit, First Fit, First Fit Decreasing (with optimal bound).
  B. Exact bin packing in CP-SAT (bin assignment booleans + load links),
     compares optimum vs FFD.
Run:  python lesson1_8c_binpacking.py
"""
import random

import numpy as np
from ortools.sat.python import cp_model

RNG = random.Random(42)
ITEMS = [RNG.randint(20, 90) for _ in range(24)]
CAP = 120


def next_fit(items, cap):
    bins, load = [], 0
    for it in items:
        if load + it > cap:
            bins.append(load)
            load = it
        else:
            load += it
    if load:
        bins.append(load)
    return len(bins)


def first_fit(items, cap):
    bins = []
    for it in items:
        for i, b in enumerate(bins):
            if b + it <= cap:
                bins[i] += it
                break
        else:
            bins.append(it)
    return len(bins)


def first_fit_decreasing(items, cap):
    return first_fit(sorted(items, reverse=True), cap)


def exact_bins(items, cap, max_bins):
    m = cp_model.CpModel()
    n = len(items)
    x = {(i, b): m.NewBoolVar(f"x{i}_{b}") for i in range(n)
         for b in range(max_bins)}
    for i in range(n):
        m.AddExactlyOne(x[i, b] for b in range(max_bins))
    for b in range(max_bins):
        m.Add(sum(items[i] * x[i, b] for i in range(n)) <= cap)
    # symmetry breaking: bin b can be used only if bin b-1 is used
    used = [m.NewBoolVar(f"u{b}") for b in range(max_bins)]
    for b in range(max_bins):
        m.AddMaxEquality(used[b], [x[i, b] for i in range(n)])
    for b in range(1, max_bins):
        m.Add(used[b] <= used[b - 1])
    m.Minimize(sum(used))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20
    solver.parameters.num_workers = 8
    st = solver.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return round(solver.ObjectiveValue()), solver.StatusName(st)
    return None, solver.StatusName(st)


def l2_lower_bound(items, cap):
    """Simple LB: total size / cap, rounded up."""
    return -(-sum(items) // cap)


if __name__ == "__main__":
    print(f"bin packing: {len(ITEMS)} items, capacity {CAP}, "
          f"total size {sum(ITEMS)}")
    print(f"  L2 lower bound (size/cap)   : {l2_lower_bound(ITEMS, CAP)}")
    print(f"  Next Fit                    : {next_fit(ITEMS, CAP)}")
    print(f"  First Fit                   : {first_fit(ITEMS, CAP)}")
    print(f"  First Fit Decreasing        : {first_fit_decreasing(ITEMS, CAP)}")
    opt, st = exact_bins(ITEMS, CAP, max_bins=first_fit(ITEMS, CAP))
    print(f"  CP-SAT exact ({st:8s}) : {opt}")
    ffd = first_fit_decreasing(ITEMS, CAP)
    if opt:
        print(f"  FFD vs optimum              : {ffd} vs {opt} "
              f"({100*(ffd-opt)/opt:+.1f}%)")
