"""Lesson 1.7 figure: CP-SAT vs Routing GLS across problem size.

Re-runs the same seeded experiment as phase1_milp/lesson1_7b (TSP family,
n = 15/30/60/120), reports objective quality and wall time, and renders:
  left  — objective values (CP-SAT exact vs GLS anytime)
  right — wall time to that answer
Run:  python phase2_mdp/make_figures_17.py
"""
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.sat.python import cp_model

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "phase1")
os.makedirs(OUT, exist_ok=True)


def make_coords(n, seed=5):
    rng = random.Random(seed)
    return [(0, 0)] + [(rng.randint(0, 120), rng.randint(0, 120))
                       for _ in range(n - 1)]


def dist(coords):
    return [[round(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
             for b in coords] for a in coords]


def solve_tsp_cpsat(M, limit_s=30.0):
    n = len(M)
    model = cp_model.CpModel()
    arcs = {(i, j): model.NewBoolVar(f"a{i}_{j}")
            for i in range(n) for j in range(n) if i != j}
    model.AddCircuit([(i, j, lit) for (i, j), lit in arcs.items()])
    model.Minimize(sum(M[i][j] * lit for (i, j), lit in arcs.items()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit_s
    solver.parameters.num_workers = 8
    t0 = time.time()
    st = solver.Solve(model)
    dt = time.time() - t0
    return (solver.ObjectiveValue() if st in (cp_model.OPTIMAL,
                                               cp_model.FEASIBLE) else None,
            dt, solver.StatusName(st))


def solve_tsp_routing(M, limit_s=5.0):
    n = len(M)
    m = pywrapcp.RoutingIndexManager(n, 1, 0)
    r = pywrapcp.RoutingModel(m)
    t = r.RegisterTransitCallback(
        lambda i, j: M[m.IndexToNode(i)][m.IndexToNode(j)])
    r.SetArcCostEvaluatorOfAllVehicles(t)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = \
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = \
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromSeconds(int(limit_s))
    t0 = time.time()
    s = r.SolveWithParameters(p)
    dt = time.time() - t0
    return (s.ObjectiveValue() if s else None), dt, "anytime GLS"


def main():
    ns = (15, 30, 60, 120)
    data = {"cpsat_v": [], "cpsat_t": [], "cpsat_st": [],
            "gls_v": [], "gls_t": []}
    for n in ns:
        M = dist(make_coords(n))
        v1, t1, st1 = solve_tsp_cpsat(M)
        v2, t2, _ = solve_tsp_routing(M)
        data["cpsat_v"].append(v1)
        data["cpsat_t"].append(t1)
        data["cpsat_st"].append(st1)
        data["gls_v"].append(v2)
        data["gls_t"].append(t2)
        print(f"n={n}: CP-SAT {v1:.0f} in {t1:.1f}s [{st1}] | "
              f"GLS {v2:.0f} in {t2:.1f}s")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    x = np.arange(len(ns))
    ax = axes[0]
    ax.plot(x, data["cpsat_v"], "o-", color="#08519c", lw=1.6,
            label="CP-SAT (exact model, 30s cap)")
    ax.plot(x, data["gls_v"], "s--", color="#d94801", lw=1.6,
            label="Routing GLS (5s anytime)")
    for i, st in enumerate(data["cpsat_st"]):
        if st != "OPTIMAL":
            ax.annotate(st.split()[0], (x[i], data["cpsat_v"][i]),
                        textcoords="offset points", xytext=(6, 8),
                        fontsize=7, color="#08519c")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("TSP nodes")
    ax.set_ylabel("tour cost (lower = better)")
    ax.set_title("solution quality across scale")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(x, data["cpsat_t"], "o-", color="#08519c", lw=1.6,
            label="CP-SAT wall time")
    ax.plot(x, data["gls_t"], "s--", color="#d94801", lw=1.6,
            label="GLS wall time (time-limited)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("TSP nodes")
    ax.set_ylabel("wall time s")
    ax.set_yscale("log")
    ax.set_title("cost of the answer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(OUT, "lesson1_7_scale.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
