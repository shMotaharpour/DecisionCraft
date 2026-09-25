"""Lesson 1.7a — Large Neighborhood Search: the bridge between exact & heuristic.

LNS = "fix most of the solution, re-optimize a fragment optimally, repeat."
We implement destroy-and-repair LNS on the 15-node TSP/CVRP data:
  destroy: remove a random segment of the tour
  repair:  solve the remaining sub-TSP to optimality with CP-SAT AddCircuit
This is exactly what CP-SAT workers do internally and what the Routing
library's local-search operators approximate.
Run:  python lesson1_7a_lns.py
"""
import json
import os
import random
import time

from ortools.sat.python import cp_model

RNG = random.Random(11)
HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "..", "data", "cvrp_15node.json")) as f:
    COORDS = json.load(f)["coords"]

M = [[round(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
      for b in COORDS] for a in COORDS]
N = len(COORDS)


def tour_len(t):
    return sum(M[t[i]][t[i + 1]] for i in range(len(t) - 1)) + M[t[-1]][t[0]]


def solve_subpath_exact(nodes):
    """Optimal ordering of a node subset starting/ending anywhere
    (we always reinsert between fixed neighbors), via CP-SAT circuit."""
    if len(nodes) <= 2:
        return nodes
    k = len(nodes)
    model = cp_model.CpModel()
    arcs = {}
    for i in range(k):
        for j in range(k):
            if i != j:
                lit = model.NewBoolVar(f"x{i}_{j}")
                arcs[(i, j)] = lit
    model.AddCircuit([(i, j, lit) for (i, j), lit in arcs.items()])
    model.Minimize(sum(M[nodes[i]][nodes[j]] * lit
                       for (i, j), lit in arcs.items()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return nodes
    order, cur = [], 0
    for _ in range(k):
        order.append(nodes[cur])
        nxt = next((j for (i, j), lit in arcs.items()
                    if i == cur and solver.Value(lit)), None)
        if nxt is None:
            break
        cur = nxt
    return order


def nearest_neighbor():
    un, t = set(range(1, N)), [0]
    while un:
        nxt = min(un, key=lambda c: M[t[-1]][c])
        t.append(nxt)
        un.remove(nxt)
    return t


def lns(t0_tour, budget_s=5.0, frag=6):
    t0_ = time.time()
    best = t0_tour[:]
    best_len = tour_len(best)
    cur, cur_len = best, best_len
    it = 0
    while time.time() - t0_ < budget_s:
        it += 1
        i = RNG.randrange(1, N - frag)
        fragment = cur[i:i + frag]
        rest = cur[:i] + cur[i + frag:]
        fixed = RNG.choice(range(len(rest) - 1))
        seq = solve_subpath_exact(fragment)
        cand = rest[:fixed + 1] + seq + rest[fixed + 1:]
        l = tour_len(cand)
        if l < cur_len:                    # accept if better (basic LNS)
            cur, cur_len = cand, l
            if l < best_len:
                best, best_len = cand, l
    return best, best_len, it


if __name__ == "__main__":
    start = nearest_neighbor()
    print(f"15-node TSP (depot 0), LNS with exact CP-SAT fragment repair")
    print(f"  nearest neighbor start : {tour_len(start):5d}")
    b, l, it = lns(start, budget_s=5.0)
    print(f"  after LNS 5s ({it} iters)  : {l:5d}   tour: {'-'.join(map(str, b))}")
    # compare: OR-Tools Routing GLS 5s reference from lesson 1.5 = 329
    print("  reference (1.5 GLS 5s) :   329")
    print()
    print("Note: LNS here is deliberately simple (single fragment, better-accept).")
    print("CP-SAT's internal LNS and Routing's operators destroy/repair many")
    print("fragments in parallel and accept sideways moves — that's the gap.")
