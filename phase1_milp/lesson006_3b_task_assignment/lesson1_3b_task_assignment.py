"""Lesson 1.3b — Task Assignment: the little model with big-model teeth.

Worker×task cost matrix, tasks must be covered, workers have capacity —
the Assignment / Generalized-Assignment family. Three solvers on the
SAME instance (8 workers, 12 tasks, capacity 2 = 24 slots for 12 tasks,
so plain "one worker per task" Hungarian handles the square core):
  scipy.optimize.linear_sum_assignment (Hungarian, O(n^3))
  OR-Tools pywraplp (MILP with binary vars — the general pattern)
  CP-SAT (AddExactlyOne per task)
Then two REAL-world upgrades that break the nice O(n^3) theory and show
why the MILP pattern earns its keep:
  (a) skills: some tasks need a skill the worker must have
  (b) fairness: workers can hold 2 tasks; minimize total COST first,
      then show the min-max-HOURS alternative and its cost premium
Measured takeaways: Hungarian speed vs MILP flexibility; the price of
each constraint upgrade; LP relaxation bound on the square case.

Run: python phase1_milp/lesson1_3b_task_assignment.py   (~5 s)
"""
import time

import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from scipy.optimize import linear_sum_assignment

rng = np.random.default_rng(14)
N_TASKS = 12
N_WORKERS = 8                    # more tasks than workers -> capacity 2
COST = rng.integers(3, 20, size=(N_WORKERS, N_TASKS)).astype(float)
HOURS = rng.integers(1, 8, size=N_TASKS).astype(float)   # task workloads
CAP = 2                          # max tasks per worker

# skills: each task needs one of 3 skills; each worker has 1-2 skills.
# CONSTRUCTED feasible: demand per skill <= 4 after clamping; each skill has
# 2 pinned single-skill + >=2 flexible two-skill workers.
SKILLS = rng.integers(0, 3, size=N_TASKS)
counts = {s: 0 for s in range(3)}
for t in range(N_TASKS):
    if counts[SKILLS[t]] >= 4:
        alt = [s for s in range(3) if counts[s] < 4]
        SKILLS[t] = min(alt, key=lambda s: counts[s])
    counts[SKILLS[t]] += 1
WORKER_SKILLS = {0: {0}, 1: {1}, 2: {2},          # pinned: 1 per skill
                 3: {1}, 4: {2}, 5: {0},          # pinned: 2nd per skill
                 6: {0, 2}, 7: {1, 0}}            # flexible pair workers
# Hall checkConstructed: skill s capable = pinned count + flex containing s
# s0: 1+2, s1: 2+1, s2: 2+1 >= demand (clamped <=3 tasks/skill for 8 workers)


def milp_solve(cost, skill_ok=None, cap=CAP, min_max=False, hours=None):
    """General-assignment MILP via pywraplp; returns (obj, assignment)."""
    solver = pywraplp.Solver.CreateSolver("SCIP")
    n, m = cost.shape
    x = {(w, t): solver.IntVar(0, 1, f"x_{w}_{t}")
         for w in range(n) for t in range(m)}
    for t in range(m):                                    # each task once
        solver.Add(sum(x[w, t] for w in range(n)) == 1)
    for w in range(n):                                    # capacity per worker
        solver.Add(sum(x[w, t] for t in range(m)) <= cap)
    if skill_ok is not None:                              # skill matching
        for w in range(n):
            for t in range(m):
                if not skill_ok[w][t]:
                    solver.Add(x[w, t] == 0)
    if min_max:                                           # fairness: min-max
        L = solver.NumVar(0, float(hours.sum()), "L")     # worker hours
        for w in range(n):
            solver.Add(sum(x[w, t] * hours[t] for t in range(m)) <= L)
        solver.Minimize(L)
    else:
        solver.Minimize(sum(x[w, t] * cost[w, t]
                            for w in range(n) for t in range(m)))
    st = solver.Solve()
    assert st == pywraplp.Solver.OPTIMAL
    assign = [-1] * m
    for w in range(n):
        for t in range(m):
            if x[w, t].solution_value() > 0.5:
                assign[t] = w
    return solver.Objective().Value(), assign


def cpsat_solve(cost):
    model = cp_model.CpModel()
    n, m = cost.shape
    x = {(w, t): model.NewBoolVar(f"x_{w}_{t}")
         for w in range(n) for t in range(m)}
    for t in range(m):
        model.AddExactlyOne(x[w, t] for w in range(n))
    for w in range(n):
        model.AddAtMostOne(x[w, t] for t in range(m))
    model.Minimize(sum(int(cost[w, t]) * x[w, t]
                       for w in range(n) for t in range(m)))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 8
    st = solver.Solve(model)
    assert st == cp_model.OPTIMAL
    return solver.ObjectiveValue()


def main():
    print(f"Generalized task assignment — {N_WORKERS} workers × "
          f"{N_TASKS} tasks, capacity {CAP}, skills + fairness\n")

    # 1) three solvers on the SQUARE core (tasks 0..7, one per worker);
    # milp_solve uses capacity CAP, so for the square core pass cap=1
    sq_cost = COST[:, :N_WORKERS]
    t0 = time.perf_counter()
    rows, cols = linear_sum_assignment(sq_cost)
    t_hung = time.perf_counter() - t0
    obj_hung = sq_cost[rows, cols].sum()

    t0 = time.perf_counter()
    obj_milp, assign = milp_solve(sq_cost, cap=1)
    t_milp = time.perf_counter() - t0

    t0 = time.perf_counter()
    obj_cpsat = cpsat_solve(sq_cost)
    t_cpsat = time.perf_counter() - t0

    print("square core (8×8, one task per worker):")
    print(f"  Hungarian  : obj={obj_hung:6.1f}  {t_hung*1e6:8.0f} µs")
    print(f"  MILP SCIP  : obj={obj_milp:6.1f}  {t_milp*1e3:8.2f} ms")
    print(f"  CP-SAT     : obj={obj_cpsat:6.1f}  {t_cpsat*1e3:8.2f} ms")
    assert abs(obj_hung - obj_milp) < 1e-6
    assert abs(obj_cpsat - obj_milp) < 1e-6

    # 2) full generalized problem: 12 tasks on 8 workers, capacity 2
    t0 = time.perf_counter()
    obj_gen, assign = milp_solve(COST)
    gen_t = time.perf_counter() - t0
    print(f"generalized (8 workers, {N_TASKS} tasks, cap {CAP}): "
          f"obj={obj_gen:6.1f}  {gen_t*1e3:.2f} ms")

    # 3) upgrade (a): skills — infeasible pairs removed
    skill_ok = [[SKILLS[t] in WORKER_SKILLS[w] for t in range(N_TASKS)]
                for w in range(N_WORKERS)]
    n_blocked = int(N_TASKS * N_WORKERS - sum(map(sum, skill_ok)))
    t0 = time.perf_counter()
    obj_skill, _ = milp_solve(COST, skill_ok=skill_ok)
    print(f"(a) skills ({n_blocked} blocked pairs): "
          f"obj={obj_skill:6.1f} (+{obj_skill - obj_gen:.1f} vs no-skill)  "
          f"{(time.perf_counter()-t0)*1e3:.2f} ms")

    # 4) upgrade (b): fairness — min-max worker HOURS vs min-cost plan
    t0 = time.perf_counter()
    obj_fair, assign_fair = milp_solve(COST, min_max=True, hours=HOURS)
    worst_plain = max(sum(HOURS[t] for t, w in enumerate(assign) if w == wk)
                      for wk in range(N_WORKERS))
    worst_fair = max(sum(HOURS[t] for t, w in enumerate(assign_fair)
                         if w == wk) for wk in range(N_WORKERS))
    cost_fair = sum(COST[w, t] for t, w in enumerate(assign_fair))
    print(f"(b) min-max hours: worst {worst_fair:.0f} h "
          f"(min-cost plan: {worst_plain:.0f} h); total cost "
          f"{cost_fair:.0f} (+{cost_fair - obj_gen:.1f} vs cheapest)  "
          f"{(time.perf_counter()-t0)*1e3:.2f} ms")

    # 5) LP relaxation on the square core (integral — prove it live)
    from scipy.optimize import linprog
    n, m = sq_cost.shape
    c = sq_cost.flatten()
    A_eq = np.zeros((n + m, n * m))
    for t in range(m):
        A_eq[t, t::m] = 1
    for w in range(n):
        A_eq[n + w, w * m:(w + 1) * m] = 1
    res = linprog(c, A_eq=A_eq, b_eq=np.ones(n + m), bounds=(0, 1),
                  method="highs")
    print(f"\nLP relaxation (square core) = {res.fun:.6f} vs integer "
          f"{obj_hung:.1f} -> gap {100*(obj_hung-res.fun)/obj_hung:.4f}% "
          f"(totally unimodular, proven)")

    print("\nread:")
    print(" - Hungarian is ~1000x faster but answers ONE square question;")
    print("   the MILP absorbs skills/capacity/fairness with 2-line edits.")
    print(" - every realistic upgrade left the polynomial world: capacity")
    print("   (generalized assignment, NP-hard), skills (feasibility risk),")
    print("   min-max (changed objective — the fair plan pays a measured")
    print("   cost premium).")
    print(" - square assignment's LP relaxation is EXACT (totally")
    print("   unimodular, proven): the one class where duality gives")
    print("   integer answers free — the bridge to lesson 1.4's duals.")
    print(" - this is the model under Chista's crew-to-quadrant assignment")
    print("   and every shift-scheduling core.")


if __name__ == "__main__":
    main()
