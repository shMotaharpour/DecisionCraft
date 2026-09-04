"""Lesson 1.2 — Logic → linear constraints with **OR-Tools** (SCIP).

Warehouse supplier selection with fixed costs:
    minimize   Σ pᵢ·xᵢ + Σ Fᵢ·yᵢ
    subject to Σ xᵢ = 100                (demand)
               xᵢ ≤ Uᵢ·yᵢ   ∀i          (activate + capacity, big-M)
               Σ yᵢ ≤ 2                 (at most 2 suppliers)
               xᵢ ≥ 0 ; yᵢ binary
"""
from ortools.linear_solver import pywraplp

p = [10.0, 12.0, 9.0, 11.0]
F = [100.0, 80.0, 150.0, 60.0]
U = [60.0, 50.0, 70.0, 40.0]
DEMAND = 100
MAX_SUPPLIERS = 2
n = len(p)

solver = pywraplp.Solver.CreateSolver("SCIP")
assert solver is not None

x = [solver.NumVar(0, U[i], f"x_{i+1}") for i in range(n)]
y = [solver.BoolVar(f"y_{i+1}") for i in range(n)]

solver.Add(sum(x) == DEMAND, "demand")
for i in range(n):
    solver.Add(x[i] <= U[i] * y[i], f"activate_{i+1}")
solver.Add(sum(y) <= MAX_SUPPLIERS, "at_most_2_suppliers")

solver.Minimize(sum(p[i] * x[i] + F[i] * y[i] for i in range(n)))

status = solver.Solve()
status_name = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
}.get(status, "not solved")
print("Status:", status_name)
print(f"Total cost = {solver.Objective().Value():,.2f}")
for i in range(n):
    fee = "paid" if y[i].solution_value() > 0.5 else "skipped"
    print(f"  supplier {i+1}: x = {x[i].solution_value():6.1f} units, y = {y[i].solution_value():.0f} (fee {fee})")