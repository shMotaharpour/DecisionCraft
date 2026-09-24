# Phase 1 — Lesson 1.6c: Population-Based Search — GA on the same TSP

Evidence: `docs/research/phase1_lesson6c_genetic_evidence.txt` (live run).
Demo: `phase1_milp/lesson1_6c_genetic_algorithm.py`.

Lesson 1.6's four engines are all **single-solution** searches: one tour,
one neighborhood move at a time. This lesson runs the other family —
**population-based** — on the *same* 40-node TSP, *same* 4 s budget, so
every number reads against 1.6a's table directly.

## 1. The GA parts (and why each is forced on permutation problems)

- **Order crossover (OX)**: copy a random slice of parent 1, fill the
  rest in parent 2's order. One-point crossover — fine for bit strings —
  breaks permutation validity (duplicate cities); OX is the fix. This
  was the first development bug (fixed before any run; documented).
- **Swap mutation** at rate 0.3: same operator family as the trio.
- **Tournament selection** (k=3) + **elitism 2**: the best two always
  survive — GA without elitism forgets its best solution.
- Fitness = tour length; no scaling tricks needed for a minimization
  with a bounded range.

## 2. Measured (4 s budget each, seed 7, 40 nodes)

![GA vs the trio](../../assets/phase1/lesson1_6c_ga_vs_trio.png)

*Bar comparison at the shared 4 s budget (seed 7): SA/ILS 713, GA
pop=8/32/128 clustered at 665–668, GRASP-no-local-search 768, and the
production GLS line at 639. The GA population-sweep bars are nearly
level — the population-size dial is flat on this instance.*

- SA (1.6a): 713 · ILS (1.6a): 713 · OR-Tools GLS: 639
- **GA pop=8: 665** · pop=32: 667 · pop=128: 668 (91104 / 84352 / 81664
  tour evals respectively)
- GRASP (greedy-randomized multi-start, 6 lines): 768

**The honest readings:** GA *beats* both single-solution engines here
(665 vs 713) — a population of diverging tours escapes the local-optimum
basin that traps SA/ILS at this size, and it does so without any
temperature schedule. It still loses to production GLS by 4% — a toy
GA vs 60 years of Routing engineering. **Population size barely matters**
(665/667/668 from 8→128) while evaluation count drops 10% — the search
is carried by crossover, not by crowd size. And GRASP without a local
search phase is not competitive (768) — its published strength is the
pairing with local search (exercise).

This is a *negative result* against GA folklore on two counts: bigger
populations don't help here, and "GA beats everything" is false — it
beats the toy trio, not the tuned library.

## 3. The two-line mentions (where the famous names fit)

- **GRASP** — greedy-randomized construction + local search; the
  construction half is 6 lines on this skeleton (measured above), the
  missing local-search half is exercise 3.
- **VNS** — systematically cycle through neighborhood structures
  (swap → 2-opt → or-opt) on descent failure; the skeleton already has
  all three operators, so VNS is exercise 4.
- **ACO / PSO** — deliberately *not* implemented: on discrete problems
  both lose to GLS/LNS/ALNS in practice, and their natural home is
  continuous optimization. Knowing where a famous method does NOT pay
  is part of the toolbox.

## 4. Bridge

- Upstream: 1.6a's engines and move operators (same N, M, RNG, budget).
- Downstream: 1.7 (LNS = population of one, restarted smartly — after
  this lesson the LNS kick reads as an extreme point of the
  population-size dial); 3.2's exploration lesson re-uses the same
  exploration/exploitation language in bandits.
- Exercises: (1) replace swap mutation with 2-opt mutation; does GA
  close on GLS? (2) add a local-search phase to GRASP (true GRASP);
  (3) implement VNS by cycling neighborhoods in `hill_climb`; (4) run
  the pop sweep at 20 s — does the flat ranking survive a bigger
  budget?
