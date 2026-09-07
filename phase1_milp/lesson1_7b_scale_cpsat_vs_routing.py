"""Lesson 1.7b — Scale test: same VRP family, exact CP-SAT vs Routing GLS.

Model the 15-node CVRP in CP-SAT with AddCircuit (per vehicle) and compare
wall time + objective against the Routing library's guided local search at
growing node counts (15 -> 30 -> 60 synthetic). Shows where exact proving
ends and metaheuristic anytime search begins.
Run:  python lesson1_7b_scale_cpsat_vs_routing.py
"""
import json
import os
import random
import time

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.sat.python import cp_model

HERE = os.path.dirname(os.path.abspath(__file__))


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
    arcs = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                arcs[(i, j)] = model.NewBoolVar(f"a{i}_{j}")
    model.AddCircuit([(i, j, lit) for (i, j), lit in arcs.items()])
    model.Minimize(sum(M[i][j] * lit for (i, j), lit in arcs.items()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit_s
    solver.parameters.num_workers = 8
    t0 = time.time()
    st = solver.Solve(model)
    dt = time.time() - t0
    if st == cp_model.OPTIMAL:
        return solver.ObjectiveValue(), dt, "OPTIMAL"
    if st == cp_model.FEASIBLE:
        return solver.ObjectiveValue(), dt, f"FEASIBLE (gap {solver.best_objective_bound:.0f})"
    return None, dt, solver.StatusName(st)


def solve_tsp_routing(M, limit_s=5.0):
    n = len(M)
    m = pywrapcp.RoutingIndexManager(n, 1, 0)
    r = pywrapcp.RoutingModel(m)
    t = r.RegisterTransitCallback(
        lambda i, j: M[m.IndexToNode(i)][m.IndexToNode(j)])
    r.SetArcCostEvaluatorOfAllVehicles(t)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromSeconds(int(limit_s))
    t0 = time.time()
    s = r.SolveWithParameters(p)
    dt = time.time() - t0
    return (s.ObjectiveValue() if s else None), dt, "anytime GLS"


if __name__ == "__main__":
    print(f"{'n':>4s} {'CP-SAT (30s cap)':>34s} {'Routing GLS (5s)':>22s}")
    for n in (15, 30, 60, 120):
        M = dist(make_coords(n))
        v1, t1, st1 = solve_tsp_cpsat(M)
        v2, t2, st2 = solve_tsp_routing(M)
        left = f"{v1:.0f} in {t1:5.1f}s [{st1}]" if v1 else f"none in {t1:5.1f}s [{st1}]"
        right = f"{v2:.0f} in {t2:4.1f}s" if v2 else "none"
        print(f"{n:4d} {left:>34s} {right:>22s}")
