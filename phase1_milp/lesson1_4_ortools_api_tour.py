"""Lesson 1.4 — Live tour of the OR-Tools pywraplp API.

Demonstrates one command from each of the 7 categories of
notes/phase1_lesson4_ortools_reference.md, including introspection of
available backends and duality info (DualValue / BestBound).
"""
from ortools.linear_solver import pywraplp

# ---- Category 1: creation & backend introspection
s = pywraplp.Solver.CreateSolver("SCIP")
assert s is not None
print("Backend version:", s.SolverVersion(), "| is MIP:", s.IsMip())

# ---- Category 2: variables of all three kinds
x = s.NumVar(0, 40, "x")            # continuous
n = s.IntVar(0, 5, "n")             # general integer
b = s.BoolVar("b")                  # binary
print("Num variables at start:", s.NumVariables())

# ---- Category 3: constraints (both styles: natural Add + row object)
s.Add(2 * x + 3 * n <= 60, "resource")
s.Add(n <= 4 * b, "n_needs_b")
s.Add(n >= 2 * b, "min_if_active")
row = s.Constraint(1, s.Infinity())  # manual row: x + n >= 1
row.SetCoefficient(x, 1.0)
row.SetCoefficient(n, 1.0)
print("Num constraints:", s.NumConstraints())

# ---- Category 4: objective, built incrementally
obj = s.Objective()
obj.SetCoefficient(x, 3.0)
obj.SetCoefficient(n, 4.0)
obj.SetCoefficient(b, -5.0)
obj.SetMaximization()

# ---- Category 5: control
s.SetTimeLimit(10000)  # ms
s.SuppressOutput()
status = s.Solve()
print("Status code:", status, "(OPTIMAL =", pywraplp.Solver.OPTIMAL, ")")

# ---- Category 6: querying
print(f"Objective value = {s.Objective().Value():.2f}")
print(f"Best bound      = {s.Objective().BestBound():.2f}")
print(f"x={x.solution_value():.2f}  n={n.solution_value():.0f}  b={b.solution_value():.0f}")
print(f"Wall time = {s.WallTime()} ms | iterations = {s.Iterations()}")
row_dual = row.DualValue()
print(f"DualValue of row 'x+n>=1' (LP concept; MIP may report 0): {row_dual}")

# ---- Category 7: export the model and inspect the math
lp_text = s.ExportModelAsLpFormat(False)
print("\n--- Exported LP model (first lines) ---")
print("\n".join(lp_text.splitlines()[:12]))