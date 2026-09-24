"""Lesson 1.2c — General integer variables (not just 0/1) with OR-Tools.

Miniature: a shop decides how many overtime shifts m to run today.
- each shift yields 300 profit, costs 100 (net 200 per shift)
- overtime allowed only if a binary switch y = 1 (a manager stays)
- manager cost 150, max 3 shifts
- a minimum staffing rule: if overtime runs, at least 2 shifts (m >= 2y)

The point: m is a GENERAL INTEGER variable in {0,1,2,3}, not a binary.
"""
from ortools.linear_solver import pywraplp

s = pywraplp.Solver.CreateSolver("SCIP")
assert s is not None
s.SuppressOutput()

y = s.BoolVar("overtime_allowed")
m = s.IntVar(0, 3, "num_shifts")  # general integer, domain {0,1,2,3}

s.Add(m <= 3 * y, "bigM_overtime_needs_manager")   # m > 0 implies y = 1
s.Add(m >= 2 * y, "min_two_shifts_if_active")      # y = 1 implies m >= 2

# profit: 200 net per shift, minus 150 manager fee when active
s.Maximize(200 * m - 150 * y)

assert s.Solve() == pywraplp.Solver.OPTIMAL
print(f"Objective = {s.Objective().Value():,.0f}")
print(f"  y (overtime allowed) = {y.solution_value():.0f}")
print(f"  m (number of shifts) = {m.solution_value():.0f}  <- general integer, not binary")