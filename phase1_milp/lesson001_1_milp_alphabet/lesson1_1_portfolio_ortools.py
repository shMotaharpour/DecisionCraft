"""Lesson 1.1 — Toy portfolio MILP with **OR-Tools** (MPSolver/SCIP).

Same formulation as lesson1_1_scipy.py:
    maximize   0.08*x_A + 0.12*x_B + 1000*x_C
    subject to x_A + x_B + 5000*x_C <= 10000
               x_B - 10000*y_B <= 0          (big-M switch)
               x_B - 2000*y_B  >= 0          (minimum if active)
               x_A, x_B >= 0 ; y_B, x_C binary
"""
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver("SCIP")
assert solver is not None, "SCIP backend not available"

x_A = solver.NumVar(0, solver.infinity(), "x_A")
x_B = solver.NumVar(0, solver.infinity(), "x_B")
y_B = solver.BoolVar("y_B")
x_C = solver.BoolVar("x_C")

solver.Add(x_A + x_B + 5000 * x_C <= 10000, "budget")
solver.Add(x_B <= 10000 * y_B, "bigM_switch")
solver.Add(x_B >= 2000 * y_B, "min_if_active")

solver.Maximize(0.08 * x_A + 0.12 * x_B + 1000 * x_C)

status = solver.Solve()
status_name = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE"}.get(status, "not solved")
print("Status:", status_name)
print(f"Objective = {solver.Objective().Value():,.2f}")
for v in (x_A, x_B, y_B, x_C):
    print(f"  {v.name():16s} = {v.solution_value():,.4f}")