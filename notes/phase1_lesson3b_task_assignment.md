# Phase 1 — Lesson 1.3b: Task Assignment — the little model with big-model teeth

Evidence: `docs/research/phase1_lesson3b_task_assignment_evidence.txt`
(live run). Demo: `phase1_milp/lesson1_3b_task_assignment.py`.

The Assignment Problem is the smallest model that still carries every
structural idea of Phase 1: binary decisions, cover/partition
constraints, dual prices — and it is the production core of
crew-to-shift, crew-to-quadrant (Chista), and machine-to-job planning.
This lesson runs it three ways, then upgrades it twice to show exactly
where the textbook theory stops and the MILP pattern starts.

## 1. Three solvers on the square core

8 workers × 8 tasks, one task per worker, minimize total cost:

| solver | objective | time |
|---|---|---|
| `scipy.optimize.linear_sum_assignment` (Hungarian) | 46.0 | **21 µs** |
| MILP SCIP (pywraplp binaries) | 46.0 | 15.4 ms |
| CP-SAT (AddExactlyOne/AtMostOne) | 46.0 | 13.5 ms |

Hungarian is ~1000× faster but answers **one** question: the square
one-cost-matrix problem. The MILP/CP-SAT pattern absorbs every realistic
extra with two-line edits.

## 2. Upgrade (a) — skills (generalized assignment)

Each task demands one of 3 skills; each worker has 1–2. 56 of 96
worker–task pairs are blocked. Live: objective 57 → **83** (+26, +46%)
— the solver must buy expensive capable workers. Feasibility is now a
*design* input: a badly skilled roster makes the model INFEASIBLE (we
hit this during the build — the generator needed Hall's condition;
see the evidence file for the failed first attempts).

## 3. Upgrade (b) — fairness (min-max workload)

Tasks carry 1–7 hours; workers hold up to 2 tasks. Two objectives:

| objective | worst worker load | total cost |
|---|---|---|
| min total cost | **12 h** | 57 |
| min-max hours (fair) | **7 h** | 159 (+102) |

The fair plan buys its balance at a measured premium (+179% cost) —
exactly the kind of tradeoff a manager must see as a number, not a
vibe. This is also the CMDP shadow-price story of lesson 2.10 in
deterministic clothing.

## 4. The LP relaxation is exact (proven, measured)

On the square core, the LP relaxation of the assignment ILP returns
46.000000 — **gap 0.0000%**. The constraint matrix is totally
unimodular, so every vertex is integral: the one family where LP
duality hands you integer answers for free. Generalized assignment
(capacity 2) loses this property — which is precisely why 1.4's dual
prices need branch-and-bound beyond this lesson.

## 5. Bridge

- Upstream: 1.1/1.2's binaries and cover constraints; 1.3's shared budget.
- Downstream: 1.4 (duals = worker reservation prices), 1.5 (VRP is
  assignment + routing), 2.10/4.6 (the same fairness/pricing logic
  under uncertainty), 4.4 (masking = blocked skill pairs in RL).
