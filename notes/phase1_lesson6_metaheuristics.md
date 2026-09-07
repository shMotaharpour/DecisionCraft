# Phase 1 — Lesson 1.6: Metaheuristics — from scratch, then inside OR-Tools

Evidence: `docs/research/phase1_lesson6_metaheuristic_evidence.txt` (live run,
ortools 9.15). Demos: `phase1_milp/lesson1_6a_metaheuristics.py` (from-scratch
engines on the 40-node TSP + 0/1 knapsack), `lesson1_6b_bignumber_milp.py`
(scalability experiment).

Why this lesson exists: OR-Tools gives you metaheuristics as a black box
(`LocalSearchMetaheuristic`). Here we open the box, implement each engine in
~30 lines, and measure the same tradeoffs the library makes for you.

## 1. The shared skeleton

All local-search metaheuristics share one loop:

```
current = initial_solution()
best = current
while budget remains:
    candidate = perturb(current)          # neighborhood move
    if accept(candidate, current):        # <-- the ONLY difference between them
        current = candidate
    if better(candidate, best): best = candidate
```

The acceptance rule is the algorithm's identity. Everything else (move
operators, restarts) is shared machinery.

## 2. The four engines (implemented in `lesson1_6a`)

| Engine | Accept rule | Escape mechanism |
|---|---|---|
| Hill climbing (greedy descent) | only if better | none — gets stuck |
| Simulated annealing | better always; worse with P=exp(-Δ/T) | temperature schedule |
| Tabu search | if better than current AND not tabu | tabu list forbids reversals |
| Iterated local search | perturb + re-optimize | random restarts from perturbed best |

Classic move operators for permutation problems (TSP): **swap** (exchange two
cities), **2-opt** (reverse a segment — kills crossing arcs), **or-opt**
(relocate a segment). For 0/1 problems: bit-flip (+ block flips).

Knapsack is the second testbed on purpose: it shows the same engines work on
subset selection, not just permutations.

## 3. Measured results (40-node TSP, seed 7, 4 s budget each)

From the live run (evidence file):
- Nearest-neighbor construction (no search): 869
- Hill climbing + 2-opt: 751 — fast drop, plateaus early (local optimum)
- Simulated annealing (T: 1000→0.01): 713
- Tabu search (swap, tenure 20): 701
- Iterated local search: 710
- **OR-Tools GLS 4 s: 639** — beats every from-scratch engine at equal wall
  time, because its C++ neighborhoods (OrOpt, segment moves) are far richer
  than naive swap/2-opt in Python.

**Honest lesson**: the gap is NOT the acceptance rule — it's the neighborhood
operators and the implementation language. Use OR-Tools' engine in production;
write your own only for exotic objectives.

Knapsack sanity check (SA, n=50): T0=10 → 1233 ("hill climbing in disguise"),
T0=1000 → 1326, greedy density baseline 1439. A cold SA is *worse than the
trivial greedy* — the classic tuning failure, measured.

## 4. Exact vs heuristic: the crossover experiment

`lesson1_6b` runs the same knapsack family at n = 50…6400 through
`scipy.optimize.milp` (exact, `mip_rel_gap=0`) and greedy+bit-flip local search
(0.5 s). Verified on this machine:

| n | exact (s) | heuristic gap |
|---|---|---|
| 50 | 0.02 | 0.35 % |
| 400 | 0.13 | 0.02 % |
| 1600 | 0.99 | 0.00 % |
| 6400 | 3.16 | 0.00 % (equal) |

Bonus finding that made this lesson: with the **default** `mip_rel_gap`,
scipy.milp at n=6400 stopped at 195846 (mip_gap 4.6e-05) while the heuristic
found 195855 — the "exact" solver returned a *worse* answer than a dumb local
search because it was allowed to stop early. Production rule: set
`mip_rel_gap` explicitly, never trust defaults, and read `res.mip_gap`.

Summary rule: small/structured → exact MILP with the certificate; huge or
exotic-objective → heuristic + LP-relaxation bound for a rigorous gap.

## 5. Inside OR-Tools: the knobs you actually get

- `local_search_metaheuristic` — GREEDY_DESCENT / GLS / SA / TABU / GENERIC_TABU
- `guided_local_search_lambda_coefficient` (default 0.1) — GLS penalty weight;
  raise it when the search keeps re-using the same arcs.
- `use_depth_first_search` — exhaustive, tiny models only.
- CP-SAT has no metaheuristic knob: it is exact OR LNS-driven anytime
  (`solver.parameters.num_workers`, `max_time_in_seconds`). LNS = the
  metaheuristic idea absorbed into exact solvers: fix most variables, free a
  fragment, re-optimize the fragment (that's what workers do in parallel).

## Pitfalls (verified)

1. SA temperature too low at start = hill climbing in disguise (measured:
   T0=10 on knapsack scored 1233 vs greedy's 1439 — worse than trivial).
2. Tabu list too short = cycling (2-cycle oscillation observed); too long =
   starves the search (tenure 50 worse than 20 on 40 nodes).
3. 2-opt on a 40-node TSP has ~780 neighbors per step — evaluating all every
   step in Python is the real bottleneck; use first-improvement, not
   best-improvement.
4. Integer costs matter (Routing callbacks must return int) — same rounding
   discipline applies to your own engines to keep Δ comparisons exact.
5. Knapsack greedy (density order) is NOT always optimal but is a strong warm
   start; greedy→local-search jump is bigger than local-search→SA.
6. **scipy.milp's default `mip_rel_gap` is NOT zero** — at n=6400 it stopped
   with gap 4.6e-05 and returned 195846 while local search found 195855.
   Set `options={"mip_rel_gap": 0.0}` when you need the true optimum, and
   always read `res.mip_gap`.

## Exercises

1. Add or-opt to the TSP engines; quantify the improvement of neighborhoods
   (expect SA+or-opt to close most of the gap to OR-Tools GLS).
2. Implement GLS from scratch on knapsack: penalty = λ·profit on over-used
   item patterns.
3. For n = 100/1,000/10,000 knapsacks, measure scipy.milp wall time; find the
   crossover where heuristic wins on this machine.
