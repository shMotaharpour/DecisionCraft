"""Lesson 1.4b — Live CP-SAT tour: one command from each of the 8 categories.

Scenario: a small trading desk schedules 3 daily tasks (risk review,
rebalancing, compliance check) on 2 analysts, with rules that would be
painful to express in MILP:
  - rebalancing must immediately follow risk review (precedence + logic)
  - the two tasks needing the same analyst cannot overlap (NoOverlap)
  - only one task can be scheduled in the 10:00 slot (AllDifferent-style
    via boolean logic)
  - an all-or-nothing client meeting (OptionalIntervalVar)
"""
import numpy as np
from ortools.sat.python import cp_model

m = cp_model.CpModel()

# ---- Category 1: variables
start = {t: m.NewIntVar(9, 17, f"start_{t}") for t in ["risk", "rebal", "compliance"]}
who = {t: m.NewIntVar(0, 1, f"analyst_{t}") for t in start}   # 2 analysts
run = {t: m.NewBoolVar(f"run_{t}") for t in ["risk", "rebal", "compliance"]}

# ---- Category 2: arithmetic — compliance must be >= risk_start + 2
m.Add(start["compliance"] >= start["risk"] + 2)

# ---- Category 3: logic (native, no big-M!)
m.AddImplication(run["rebal"], run["risk"])          # rebal implies risk ran
m.AddExactlyOne([run["risk"], run["compliance"]])    # one of the two, exactly
m.AddBoolOr([who["risk"].Not(), who["rebal"].Not()]) # risk and rebal use different analysts

# ---- Category 4: combinatorial — the three start times are all distinct
m.AddAllDifferent([start["risk"], start["rebal"], start["compliance"]])
# table: rebalancing analyst must differ from risk analyst (whitelist)
m.AddAllowedAssignments(
    [who["risk"], who["rebal"]],
    [(0, 1), (1, 0)],
)

# ---- Category 6: scheduling intervals
intervals = [
    m.NewOptionalIntervalVar(start["risk"], 1, start["risk"] + 1, run["risk"], "iv_risk"),
    m.NewOptionalIntervalVar(start["rebal"], 2, start["rebal"] + 2, run["rebal"], "iv_rebal"),
    m.NewIntervalVar(start["compliance"], 1, start["compliance"] + 1, "iv_comp"),
]
m.AddNoOverlap(intervals)  # one shared calendar: no two tasks overlap

# ---- Category 7: objective — do the valuable tasks, earlier is better
m.Maximize(10 * run["risk"] + 8 * run["rebal"] + 5 * run["compliance"]
           - (start["risk"] + start["rebal"] + start["compliance"] - 27) * 0.1)
m.AddHint(who["risk"], 0)

# ---- Category 8: solve with parameters
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 10
solver.parameters.num_workers = 4
status = solver.Solve(m)
print("Status:", solver.StatusName(status))
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    for t in start:
        print(f"  {t:12s} start={solver.Value(start[t])}:00 analyst={solver.Value(who[t])} runs={solver.BooleanValue(run[t])}")
    print(f"Objective = {solver.ObjectiveValue():.2f}, bound = {solver.BestObjectiveBound():.2f}")
    print(solver.ResponseStats())

# ---- Bonus: enumerate ALL solutions of a tiny subproblem
print("\n--- All solutions of x+y=1, x,y binary ---")
m2 = cp_model.CpModel()
xs, ys = m2.NewBoolVar("x"), m2.NewBoolVar("y")
m2.Add(xs + ys == 1)
solver2 = cp_model.CpSolver()
solver2.parameters.enumerate_all_solutions = True


class Collector(cp_model.CpSolverSolutionCallback):
    def __init__(self, xs, ys):
        super().__init__()
        self.seen = []

    def on_solution_callback(self):
        self.seen.append((self.BooleanValue(xs), self.BooleanValue(ys)))


cb = Collector(xs, ys)
solver2.SearchForAllSolutions(m2, cb)
print("All solutions:", cb.seen)
