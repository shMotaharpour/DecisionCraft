"""Lesson 1.6a — Metaheuristics from scratch: 4 engines, 2 testbeds.

Engines (each ~30 lines, shared skeleton, only the ACCEPT rule differs):
  hill climbing | simulated annealing | tabu search | iterated local search
Testbeds:
  A. 40-node Euclidean TSP (seed 7, same instance as lesson 1.5 part E)
  B. 0/1 knapsack (n=50, seeded) — same engines on subset selection
Finale: OR-Tools GLS on the same TSP instance at the same wall-time budget.
Run:  python lesson1_6a_metaheuristics.py
"""
import math
import random
import time

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

RNG = random.Random(7)

# ---------------------------------------------------------------- testbed A
N = 40
COORDS = [(0, 0)] + [(RNG.randint(0, 120), RNG.randint(0, 120)) for _ in range(N - 1)]
M = [[round(math.dist(a, b)) for b in COORDS] for a in COORDS]


def tour_len(t):
    return sum(M[t[i]][t[i + 1]] for i in range(len(t) - 1)) + M[t[-1]][t[0]]


def nearest_neighbor():
    un, t = set(range(1, N)), [0]
    while un:
        nxt = min(un, key=lambda c: M[t[-1]][c])
        t.append(nxt)
        un.remove(nxt)
    return t


def two_opt_neighbors(t):
    """All 2-opt moves: reverse segment [i:j]."""
    for i in range(1, len(t) - 1):
        for j in range(i + 1, len(t)):
            yield t[:i] + t[i:j][::-1] + t[j:]


def swap_neighbors(t):
    for i in range(1, len(t) - 1):
        for j in range(i + 1, len(t)):
            s = t[:]
            s[i], s[j] = s[j], s[i]
            yield s


def random_two_opt(t):
    i, j = sorted(RNG.sample(range(1, len(t)), 2))
    return t[:i] + t[i:j][::-1] + t[j:]


# ------------------------------------------------------------- engines
def hill_climb(t0, budget_s, neigh=two_opt_neighbors):
    t0_ = time.time()
    cur, cur_len = t0, tour_len(t0)
    improved = True
    while improved and time.time() - t0_ < budget_s:
        improved = False
        for cand in neigh(cur):
            l = tour_len(cand)
            if l < cur_len:
                cur, cur_len, improved = cand, l, True
                break                      # first-improvement
    return cur, cur_len


def simulated_annealing(t0, budget_s, T0=1000.0, Tend=0.01, alpha=0.9995):
    t0_ = time.time()
    cur, cur_len = t0, tour_len(t0)
    best, best_len = cur, cur_len
    T = T0
    while time.time() - t0_ < budget_s:
        cand = random_two_opt(cur)
        l = tour_len(cand)
        d = l - cur_len
        if d <= 0 or RNG.random() < math.exp(-d / max(T, 1e-12)):
            cur, cur_len = cand, l
            if l < best_len:
                best, best_len = cand, l
        T = max(T * alpha, Tend)
    return best, best_len


def tabu_search(t0, budget_s, tenure=20):
    t0_ = time.time()
    cur, cur_len = t0, tour_len(t0)
    best, best_len = cur, cur_len
    tabu = []                          # list of frozen (i,j) move keys
    while time.time() - t0_ < budget_s:
        best_c, best_l, best_mv = None, math.inf, None
        for i in range(1, len(cur) - 1):
            for j in range(i + 1, len(cur)):
                if (i, j) in tabu:
                    continue
                s = cur[:]
                s[i], s[j] = s[j], s[i]
                l = tour_len(s)
                if l < best_l:
                    best_c, best_l, best_mv = s, l, (i, j)
        if best_c is None:
            break
        cur, cur_len = best_c, best_l
        tabu.append(best_mv)
        if len(tabu) > tenure:
            tabu.pop(0)
        if best_l < best_len:
            best, best_len = best_c, best_l
    return best, best_len


def iterated_local_search(t0, budget_s, kicks=50):
    t0_ = time.time()
    best, best_len = hill_climb(t0, budget_s / kicks / 2)
    while time.time() - t0_ < budget_s:
        kicked = random_two_opt(best)
        cand, l = hill_climb(kicked, budget_s / kicks / 2)
        if l < best_len:
            best, best_len = cand, l
    return best, best_len


# ---------------------------------------------------------------- testbed B
KNAP_W = [RNG.randint(2, 40) for _ in range(50)]
KNAP_P = [RNG.randint(5, 90) for _ in range(50)]
KNAP_C = sum(KNAP_W) // 3


def knap_len(bits):
    w = sum(w for w, b in zip(KNAP_W, bits) if b)
    p = sum(p for p, b in zip(KNAP_P, bits) if b)
    return -p if w <= KNAP_C else 10**6 + (w - KNAP_C) * 100


def knap_neighbor(bits):
    i = RNG.randrange(len(bits))
    s = bits[:]
    s[i] = 1 - s[i]
    return s


def greedy_knap():
    order = sorted(range(50), key=lambda i: -KNAP_P[i] / KNAP_W[i])
    bits, w = [0] * 50, 0
    for i in order:
        if w + KNAP_W[i] <= KNAP_C:
            bits[i], w = 1, w + KNAP_W[i]
    return bits


def sa_knapsack(budget_s, T0):
    t0_ = time.time()
    cur = [0] * 50
    cur_len = knap_len(cur)
    best, best_len = cur, cur_len
    T = T0
    while time.time() - t0_ < budget_s:
        cand = knap_neighbor(cur)
        l = knap_len(cand)
        d = l - cur_len
        if d <= 0 or RNG.random() < math.exp(-d / max(T, 1e-12)):
            cur, cur_len = cand, l
            if l < best_len:
                best, best_len = cand, l
        T = max(T * 0.999, 0.01)
    return best_len


def or_tools_gls_tsp(seconds=4):
    m = pywrapcp.RoutingIndexManager(N, 1, 0)
    r = pywrapcp.RoutingModel(m)
    t = r.RegisterTransitCallback(lambda i, j: M[m.IndexToNode(i)][m.IndexToNode(j)])
    r.SetArcCostEvaluatorOfAllVehicles(t)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromSeconds(seconds)
    s = r.SolveWithParameters(p)
    return s.ObjectiveValue() if s else None


if __name__ == "__main__":
    BUDGET = 4.0
    print(f"40-node TSP (seed 7), budget {BUDGET}s each, 2-opt moves")
    print(f"  nearest-neighbor construction : {tour_len(nearest_neighbor()):6d}")
    t, l = hill_climb(nearest_neighbor(), BUDGET)
    print(f"  hill climbing (2-opt, 1st-im) : {l:6d}")
    _, l = simulated_annealing(nearest_neighbor(), BUDGET)
    print(f"  simulated annealing           : {l:6d}")
    _, l = tabu_search(nearest_neighbor(), BUDGET)
    print(f"  tabu search (swap, tenure 20) : {l:6d}")
    _, l = iterated_local_search(nearest_neighbor(), BUDGET)
    print(f"  iterated local search         : {l:6d}")
    print(f"  OR-Tools GLS {BUDGET:.0f}s             : {or_tools_gls_tsp(int(BUDGET)):6d}")

    print()
    print("0/1 knapsack n=50, capacity = W/3, SA at two starting temperatures")
    print(f"  greedy density baseline       : {-knap_len(greedy_knap()):6d}")
    print(f"  SA  T0=10   (too cold)        : {-sa_knapsack(2.0, 10.0):6d}")
    print(f"  SA  T0=1000 (proper)          : {-sa_knapsack(2.0, 1000.0):6d}")
