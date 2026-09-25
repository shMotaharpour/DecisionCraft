"""Lesson 1.1 — Toy portfolio MILP with **scipy.optimize.milp**.

Same formulation as lesson1_1_portfolio_ortools.py:
    maximize   0.08*x_A + 0.12*x_B + 1000*x_C
    subject to x_A + x_B + 5000*x_C <= 10000
               x_B - 10000*y_B <= 0          (big-M switch)
               x_B - 2000*y_B  >= 0          (minimum if active)
               x_A, x_B >= 0 ; y_B, x_C binary

scipy.optimize.milp uses HiGHS under the hood; the model is given in
matrix form A_ub x <= b_ub. We negate rows that need a >= lower bound
(x_B - 2000*y_B >= 0  becomes  -(x_B - 2000*y_B) <= 0).
"""
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

c = np.array([-0.08, -0.12, 0.0, -1000.0])  # scipy minimizes -> negate objective

A_ub = np.array([
    [1.0, 1.0, 0.0, 5000.0],   # budget:  x_A + x_B + 5000*x_C <= 10000
    [0.0, 1.0, -10000.0, 0.0],  # big-M:   x_B - 10000*y_B <= 0
    [0.0, -1.0, 2000.0, 0.0],   # min:   -(x_B - 2000*y_B) <= 0
])
b_ub = np.array([10000.0, 0.0, 0.0])

integrality = np.array([0, 0, 1, 1])  # 0=continuous, 1=integer (0/1 bounds make binary)

res = milp(
    c=c,
    constraints=LinearConstraint(A_ub, -np.inf, b_ub),
    bounds=Bounds(lb=[0, 0, 0, 0], ub=[np.inf, np.inf, 1, 1]),
    integrality=integrality,
)

print("Status:", res.message)
print(f"Objective = {-res.fun:,.2f}")
names = ["x_A", "x_B", "y_B (B switch)", "x_C (invest C)"]
for name, val in zip(names, res.x):
    print(f"  {name:16s} = {val:,.4f}")