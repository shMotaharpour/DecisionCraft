"""Lesson 1.8b — Network flows: min-cost flow, max flow, assignment.

Three miniatures with the dedicated graph solvers (NOT CP-SAT):
  A. Min-cost flow — transport: 3 plants -> 4 warehouses, cheapest routing.
  B. Max flow — pipeline capacity: how much can flow from s to t?
  C. Assignment as min-cost flow — 5 workers -> 5 tasks, one task each.
Run:  python lesson1_8b_flows.py
"""
from ortools.graph.python import max_flow, min_cost_flow

# ------------------------------------------------------------------ A
PLANTS = {"P1": 30, "P2": 25, "P3": 20}
WAREHOUSES = {"W1": 20, "W2": 15, "W3": 25, "W4": 15}
# unit shipping cost plant -> warehouse (row per plant)
COST = [[8, 6, 10, 9],
        [9, 12, 7, 6],
        [5, 7, 8, 10]]


def part_a_min_cost():
    names = list(PLANTS) + list(WAREHOUSES)
    idx = {n: i for i, n in enumerate(names)}
    src, snk = len(names), len(names) + 1
    s = min_cost_flow.SimpleMinCostFlow()
    for p, sup in PLANTS.items():
        s.add_arc_with_capacity_and_unit_cost(src, idx[p], sup, 0)
    for w, dem in WAREHOUSES.items():
        s.add_arc_with_capacity_and_unit_cost(idx[w], snk, dem, 0)
    for i, p in enumerate(PLANTS):
        for j, w in enumerate(WAREHOUSES):
            s.add_arc_with_capacity_and_unit_cost(idx[p], idx[w], 100,
                                                  COST[i][j])
    s.set_node_supply(src, sum(PLANTS.values()))
    s.set_node_supply(snk, -sum(WAREHOUSES.values()))
    st = s.solve()
    status_name = {s.OPTIMAL: "OPTIMAL", s.FEASIBLE: "FEASIBLE",
                   s.INFEASIBLE: "INFEASIBLE"}.get(st, f"status {st}")
    print(f"A. min-cost transport: {status_name}, "
          f"cost = {s.optimal_cost()}")
    for arc in range(s.num_arcs()):
        f = s.flow(arc)
        if f and s.tail(arc) != src and s.head(arc) != snk:
            print(f"   {names[s.tail(arc)]} -> {names[s.head(arc)]}: {f}")
    print(f"   supplies sum {sum(PLANTS.values())}, "
          f"demands sum {sum(WAREHOUSES.values())}")


# ------------------------------------------------------------------ B
def part_b_max_flow():
    """Nodes 0..5; max flow from 0 to 5."""
    caps = [(0, 1, 10), (0, 2, 8), (1, 3, 5), (1, 4, 5), (2, 3, 5),
            (2, 5, 8), (3, 5, 6), (4, 5, 10)]
    s = max_flow.SimpleMaxFlow()
    for u, v, c in caps:
        s.add_arc_with_capacity(u, v, c)
    st = s.solve(0, 5)
    status_name = {s.OPTIMAL: "OPTIMAL"}.get(st, f"status {st}")
    print(f"B. max flow 0->5: {status_name} = {s.optimal_flow()}")
    for i in range(s.num_arcs()):
        if s.flow(i):
            print(f"   {s.tail(i)} -> {s.head(i)}: {s.flow(i)}/{caps[i][2]}")


# ------------------------------------------------------------------ C
def part_c_assignment():
    costs = [[9, 2, 7, 8, 6],
             [6, 4, 3, 7, 5],
             [5, 8, 1, 8, 8],
             [7, 6, 9, 4, 2],
             [8, 2, 6, 5, 4]]
    n = len(costs)
    s = min_cost_flow.SimpleMinCostFlow()
    src, snk = 2 * n, 2 * n + 1
    for w in range(n):
        s.add_arc_with_capacity_and_unit_cost(src, w, 1, 0)
    for t in range(n, 2 * n):
        s.add_arc_with_capacity_and_unit_cost(t, snk, 1, 0)
    for w in range(n):
        for t in range(n, 2 * n):
            s.add_arc_with_capacity_and_unit_cost(w, t, 1, costs[w][t - n])
    s.set_node_supply(src, n)
    s.set_node_supply(snk, -n)
    st = s.solve()
    status_name = {s.OPTIMAL: "OPTIMAL", s.FEASIBLE: "FEASIBLE",
                   s.INFEASIBLE: "INFEASIBLE"}.get(st, f"status {st}")
    print(f"C. assignment: {status_name}, total cost = {s.optimal_cost()}")
    for arc in range(s.num_arcs()):
        if (s.flow(arc) and n <= s.tail(arc) < 2 * n
                and s.head(arc) != snk):
            w, t = s.tail(arc), s.head(arc) - n
            print(f"   worker {w} -> task {t} (cost {costs[w][t]})")


if __name__ == "__main__":
    part_a_min_cost()
    print()
    part_b_max_flow()
    print()
    part_c_assignment()
