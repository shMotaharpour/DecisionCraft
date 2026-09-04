"""Lesson 1.2 — Logic → linear constraints with **scipy.optimize.milp**.

Warehouse supplier selection with fixed costs:
    minimize   Σ pᵢ·xᵢ + Σ Fᵢ·yᵢ
    subject to Σ xᵢ = 100                (demand)
               xᵢ ≤ Uᵢ·yᵢ   ∀i          (activate + capacity, big-M)
               Σ yᵢ ≤ 2                 (at most 2 suppliers)
               xᵢ ≥ 0 ; yᵢ binary
"""
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

p = np.array([10.0, 12.0, 9.0, 11.0])
F = np.array([100.0, 80.0, 150.0, 60.0])
U = np.array([60.0, 50.0, 70.0, 40.0])
DEMAND = 100.0
MAX_SUPPLIERS = 2
n = len(p)

# Variables: x0..x3 (units), y0..y3 (activation binaries)
c = np.concatenate([p, F])  # scipy minimizes, objective is already a cost

# Equality demand: Σ xᵢ = 100  -> 1*x + 0*y = 100
A_eq = np.zeros((1, 2 * n))
A_eq[0, :n] = 1.0

# Activation: xᵢ - Uᵢ·yᵢ ≤ 0  for each i
A_act = np.zeros((n, 2 * n))
for i in range(n):
    A_act[i, i] = 1.0
    A_act[i, n + i] = -U[i]

# At most 2 suppliers: Σ yᵢ ≤ 2
A_card = np.zeros((1, 2 * n))
A_card[0, n:] = 1.0

A = np.vstack([A_act, A_card])
b = np.concatenate([-np.zeros(n), [MAX_SUPPLIERS]])  # activation upper bounds 0, card 2

res = milp(
    c=c,
    constraints=[
        LinearConstraint(A_eq, lb=DEMAND, ub=DEMAND),
        LinearConstraint(A, lb=-np.inf, ub=b),
    ],
    bounds=Bounds(
        lb=np.concatenate([np.zeros(n), np.zeros(n)]),
        ub=np.concatenate([U, np.ones(n)]),
    ),
    integrality=np.concatenate([np.zeros(n), np.ones(n)]),
)

print("Status:", res.message)
print(f"Total cost = {-(-res.fun):,.2f}" if res.fun <= 0 else f"Total cost = {res.fun:,.2f}")
for i in range(n):
    print(f"  supplier {i+1}: x = {res.x[i]:6.1f} units, y = {res.x[n+i]:.0f} (fee {'paid' if res.x[n+i] > 0.5 else 'skipped'})")