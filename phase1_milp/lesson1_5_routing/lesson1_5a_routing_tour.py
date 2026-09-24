"""Lesson 1.5a — Routing library tour: TSP -> CVRP -> VRPTW -> P&D (OR-Tools).

Four miniatures in one file, all on the same 15-node instance:
  A. TSP   — 1 vehicle, all nodes, arc cost = distance
  B. CVRP  — K vehicles + capacity dimension
  C. VRPTW — + time dimension with windows
  D. P&D   — pickup & delivery pairs (order + same-vehicle enforced)
Also enumerates every FirstSolutionStrategy / LocalSearchMetaheuristic value.
Run:  python lesson1_5a_routing_tour.py
"""
import json
import os

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "cvrp_15node.json")

First = routing_enums_pb2.FirstSolutionStrategy
Meta = routing_enums_pb2.LocalSearchMetaheuristic


def load():
    with open(DATA) as f:
        d = json.load(f)
    return d["coords"], d["demands"], d["pickup_pairs"]


def dist_matrix(coords):
    n = len(coords)
    return [[round(((coords[i][0] - coords[j][0]) ** 2 +
                    (coords[i][1] - coords[j][1]) ** 2) ** 0.5)
             for j in range(n)] for i in range(n)]


def build_routing(n, k, depot, cost_cb, scale=1):
    manager = pywrapcp.RoutingIndexManager(n, k, depot)
    routing = pywrapcp.RoutingModel(manager)
    transit = routing.RegisterTransitCallback(
        lambda fi, ti: scale * cost_cb(manager.IndexToNode(fi),
                                       manager.IndexToNode(ti)))
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    return manager, routing


def solve(manager, routing, meta=Meta.GUIDED_LOCAL_SEARCH, seconds=3):
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = First.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = meta
    p.time_limit.FromSeconds(seconds)
    return routing.SolveWithParameters(p)


def extract(manager, routing, sol, k=0):
    route, i = [], routing.Start(k)
    while not routing.IsEnd(i):
        route.append(manager.IndexToNode(i))
        i = sol.Value(routing.NextVar(i))
    route.append(manager.IndexToNode(i))
    return route


def print_route(route):
    return " -> ".join(str(x) for x in route)


def part_a_tsp(coords, M):
    print("=" * 62)
    print("A. TSP — 1 vehicle, all cities + depot 0")
    n = len(M)
    manager, routing = build_routing(n, 1, 0,
                                     lambda a, b: M[a][b])
    sol = solve(manager, routing)
    route = extract(manager, routing, sol)
    cost = sum(M[route[i]][route[i + 1]] for i in range(len(route) - 1))
    print(f"   route ({len(route)-2} cities): {print_route(route)}")
    print(f"   total cost {cost}")


def part_b_cvrp(M, demands, k=2, cap=100):
    print("=" * 62)
    print(f"B. CVRP — {k} vehicles, capacity {cap}")
    n = len(M)
    manager, routing = build_routing(n, k, 0, lambda a, b: M[a][b])
    def demand_cb(fi):
        return demands[manager.IndexToNode(fi)]
    dem_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(dem_idx, 0, [cap] * k,
                                            True, "Capacity")
    sol = solve(manager, routing)
    for v in range(k):
        r = extract(manager, routing, sol, v)
        load = sum(demands[x] for x in r if x)
        print(f"   vehicle {v}: {print_route(r)}  load {load}")
    print(f"   objective {sol.ObjectiveValue()}")


def part_c_vrptw(M, demands, service=5):
    print("=" * 62)
    print("C. VRPTW — + time windows (horizon 300, service", service, "min)")
    n = len(M)
    K, cap = 2, 100
    horizon = 300
    manager = pywrapcp.RoutingIndexManager(n, K, 0)
    routing = pywrapcp.RoutingModel(manager)
    dist_idx = routing.RegisterTransitCallback(
        lambda fi, ti: M[manager.IndexToNode(fi)][manager.IndexToNode(ti)])
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    def time_cb(fi, ti):
        a, b = manager.IndexToNode(fi), manager.IndexToNode(ti)
        return service + M[a][b]
    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_idx, 10, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    # Window derived from straight-line depot distance => always satisfiable;
    # every 4th node gets a tighter (more binding) window.
    for node in range(1, n):
        d0 = M[0][node]
        if node % 4 == 0:
            lo, hi = int(d0 * 0.8), int(d0 * 1.6) + 30
        else:
            lo, hi = 0, horizon
        time_dim.CumulVar(node).SetRange(lo, hi)
    for v in range(K):
        routing.AddVariableMinimizedByFinalizer(
            time_dim.CumulVar(routing.End(v)))
    sol = solve(manager, routing)
    if sol:
        for v in range(K):
            r = extract(manager, routing, sol, v)
            print(f"   vehicle {v}: {print_route(r)}")
        print(f"   objective {sol.ObjectiveValue()}")
    else:
        print("   no solution within time limit")


def part_d_pickup_delivery(M, pairs):
    print("=" * 62)
    print(f"D. Pickup & delivery — pairs {pairs}")
    K = 2
    n = len(M)
    manager = pywrapcp.RoutingIndexManager(n, K, 0)
    routing = pywrapcp.RoutingModel(manager)
    dist_idx = routing.RegisterTransitCallback(
        lambda fi, ti: M[manager.IndexToNode(fi)][manager.IndexToNode(ti)])
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)
    routing.AddDimension(dist_idx, 0, 3000, True, "Distance")
    dist_dim = routing.GetDimensionOrDie("Distance")
    for a, b in pairs:
        routing.AddPickupAndDelivery(a, b)
        routing.solver().Add(
            routing.VehicleVar(a) == routing.VehicleVar(b))
        routing.solver().Add(
            dist_dim.CumulVar(a) <= dist_dim.CumulVar(b))
    sol = solve(manager, routing)
    if sol:
        for v in range(K):
            print(f"   vehicle {v}: {print_route(extract(manager, routing, sol, v))}")
        print(f"   objective {sol.ObjectiveValue()}")


def part_e_enumerate_strategies(M, seed=7):
    print("=" * 62)
    print("E. First-solution strategy x metaheuristic — random 40-node TSP")
    print("   (15 nodes is so small that everything converges to 329; we")
    print("    need a bigger instance to SEE the strategy differences)")
    import random as _r
    rng = _r.Random(seed)
    coords40 = [(0, 0)] + [(rng.randint(0, 120), rng.randint(0, 120))
                           for _ in range(39)]
    M40 = dist_matrix(coords40)
    n = len(M40)
    strategies = [First.AUTOMATIC, First.PATH_CHEAPEST_ARC,
                  First.PARALLEL_CHEAPEST_INSERTION, First.LOCAL_CHEAPEST_INSERTION,
                  First.BEST_INSERTION, First.CHRISTOFIDES, First.SWEEP,
                  First.GLOBAL_CHEAPEST_ARC, First.FIRST_UNBOUND_MIN_VALUE]
    metas = [Meta.GREEDY_DESCENT, Meta.GUIDED_LOCAL_SEARCH,
             Meta.SIMULATED_ANNEALING, Meta.TABU_SEARCH,
             Meta.GENERIC_TABU_SEARCH]
    header = " ".join(f"{n:>10s}" for n in
                      ("GRDSC", "GLS", "SIMANN", "TABU", "GTABU"))
    print(f"   {'strategy':30s} {header}")
    NAMES = {First.AUTOMATIC: "AUTOMATIC",
             First.PATH_CHEAPEST_ARC: "PATH_CHEAPEST_ARC",
             First.PARALLEL_CHEAPEST_INSERTION: "PARALLEL_CHEAPEST_INS",
             First.LOCAL_CHEAPEST_INSERTION: "LOCAL_CHEAPEST_INS",
             First.BEST_INSERTION: "BEST_INSERTION",
             First.CHRISTOFIDES: "CHRISTOFIDES",
             First.GLOBAL_CHEAPEST_ARC: "GLOBAL_CHEAPEST_ARC",
             First.FIRST_UNBOUND_MIN_VALUE: "FIRST_UNBOUND_MIN_VAL"}
    # NOTE: SWEEP dropped — needs a sweep arranger that is undefined for plain
    # 2-D models in recent OR-Tools ("Undefined sweep arranger" error at solve
    # time). Mention in the note, don't crash the demo.
    for s in [x for x in strategies if x in NAMES]:
        row = []
        for m in metas:
            manager, routing = build_routing(n, 1, 0, lambda a, b: M40[a][b])
            p = pywrapcp.DefaultRoutingSearchParameters()
            p.first_solution_strategy = s
            p.local_search_metaheuristic = m
            p.time_limit.FromSeconds(2)
            sol = routing.SolveWithParameters(p)
            row.append(sol.ObjectiveValue() if sol else None)
        cells = " ".join(f"{c:10d}" if c is not None else "    none  "
                         for c in row)
        print(f"   {NAMES[s]:30s} {cells}")


if __name__ == "__main__":
    coords, demands, pickup_pairs = load()
    M = dist_matrix(coords)
    part_a_tsp(coords, M)
    part_b_cvrp(M, demands)
    part_c_vrptw(M, demands)
    part_d_pickup_delivery(M, pickup_pairs)
    part_e_enumerate_strategies(M)
