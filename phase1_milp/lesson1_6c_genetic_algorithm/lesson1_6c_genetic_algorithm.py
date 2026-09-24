"""Lesson 1.6c — Population-based: Genetic Algorithm vs the single-solution trio.

Same 40-node TSP (seed 7), same per-engine budget, same 2-opt move family
as lesson 1.6a — so every number is directly comparable. Adds:
  GA with order-crossover (OX) + swap mutation + tournament selection
  + elitism; population sweep {8, 32, 128} at the SAME total budget to
  show the exploration/exploitation dial measured, not asserted.
  Plus the two-line mentions: GRASP and VNS as exercises on this exact
  skeleton (where they fit, where they don't).
Honest expectation to verify: GA pays for diversity; on permutation TSP
with 2-opt neighborhoods ILS usually wins at equal budget. If it does,
the table says so (negative results are the course style).

Run: python phase1_milp/lesson1_6c_genetic_algorithm.py   (~20 s)
"""
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesson1_6a_metaheuristics import (N, M, RNG, nearest_neighbor,
                                       iterated_local_search,
                                       simulated_annealing, tour_len)
from lesson1_6a_metaheuristics import or_tools_gls_tsp


def ox_crossover(p1, p2):
    """Order crossover (OX): copy a random slice of p1, fill the rest in
    p2's order — preserves permutation validity, the reason TSP-GA uses
    OX instead of one-point crossover (which would duplicate cities)."""
    n = len(p1)
    i, j = sorted(RNG.sample(range(n), 2))
    child = [None] * n
    child[i:j + 1] = p1[i:j + 1]
    fill = [c for c in p2 if c not in child]
    k = 0
    for idx in list(range(j + 1, n)) + list(range(i)):
        child[idx] = fill[k]
        k += 1
    return tuple(child)


def swap_mutate(t):
    i, j = sorted(RNG.sample(range(len(t)), 2))
    t = list(t)
    t[i], t[j] = t[j], t[i]
    return tuple(t)


def tournament(pop, fits, k=3):
    idxs = RNG.sample(range(len(pop)), k)
    return pop[min(idxs, key=lambda i: fits[i])]


def ga(t0, budget_s, pop_size=32, p_mut=0.3, elite=2):
    t_start = time.time()
    pop = [tuple(RNG.sample(range(N), N)) for _ in range(pop_size - 1)]
    pop.append(tuple(t0))
    fits = [tour_len(t) for t in pop]
    best = min(fits)
    evals = pop_size
    while time.time() - t_start < budget_s - evals * 2.2e-5:
        newpop = sorted(range(pop_size), key=lambda i: fits[i])[:elite]
        newpop = [pop[i] for i in newpop]
        while len(newpop) < pop_size:
            c = ox_crossover(tournament(pop, fits), tournament(pop, fits))
            if RNG.random() < p_mut:
                c = swap_mutate(c)
            newpop.append(c)
        pop = newpop
        fits = [tour_len(t) for t in pop]
        evals += pop_size
        best = min(best, min(fits))
    return int(best), evals


def main():
    BUDGET = 4.0
    nn = nearest_neighbor()
    print(f"40-node TSP (seed 7), budget {BUDGET}s per engine — "
          f"directly comparable to lesson 1.6a\n")
    t, l = simulated_annealing(nn, BUDGET)
    print(f"  simulated annealing (1.6a)        : {l:6d}")
    t, l = iterated_local_search(nn, BUDGET)
    print(f"  iterated local search (1.6a)      : {l:6d}")
    print(f"  OR-Tools GLS (1.6a)               : "
          f"{or_tools_gls_tsp(int(BUDGET)):6d}")

    print("\nGA population sweep (same 4 s total budget, OX + swap-mut + "
          "tournament + elite-2):")
    for pop_size in (8, 32, 128):
        best, evals = ga(nn, BUDGET, pop_size=pop_size)
        print(f"  GA pop={pop_size:4d}                    : {best:6d}  "
              f"({evals} tour evals)")

    # two-line mentions, implemented as one-liners on this skeleton
    # GRASP = greedy-randomized multi-start (alpha-restricted candidate list)
    def grasp(budget_s, alpha=0.15, repeats=None):
        t_s = time.time()
        best = math.inf
        while time.time() - t_s < budget_s:
            t = [RNG.randrange(N)]
            while len(t) < N:
                cands = sorted((M[t[-1]][c], c) for c in range(N)
                               if c not in t)
                k = max(1, int(alpha * len(cands)))
                t.append(cands[RNG.randrange(k)][1])
            best = min(best, tour_len(tuple(t)))
        return int(best)

    print(f"\n  GRASP (greedy-randomized multistart): "
          f"{grasp(BUDGET):6d}   <- 6 lines on this skeleton; VNS = "
          f"cycle neighborhoods, see exercises")


if __name__ == "__main__":
    main()
