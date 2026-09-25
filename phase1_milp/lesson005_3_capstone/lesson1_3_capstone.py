"""Lesson 1.3 Capstone — Joint Portfolio Allocation + Warehouse Network.

One MILP, two coupled subproblems sharing a single capital budget.

Portfolio (invest x_i dollars in asset i, i=1..5):
  returns r = [0.05, 0.08, 0.12, 0.10, 0.15]
  risk   q = [0.5,  1.0,  2.0,  1.5,  3.0]
  - total risk:       sum(q_i * x_i) <= 8.0     (x in thousands of $)
  - sector cap:       x3 + x4 <= 0.40 * (sum of all x)
  - fund 5 all-or-nothing ticket: x5 >= 3 * y5  (y5 binary), x5 <= 20 * y5
  - at most 4 assets: sum_i y_i^p <= 4  (activation binary per asset)

Warehouse (open w in {1,2,3}, serve 3 regions with total demand D=100):
  fixed cost F = [120, 90, 150] (thousand $)
  capacity   C = [60, 50, 70]
  flow_{w,j} >= 0: units shipped from warehouse w to region j; demand_j sums to D
  - demand:      sum_w flow_{w,j} = d_j
  - capacity:    sum_j flow_{w,j} <= C_w * z_w       (z_w binary)
  - min share:   sum_j flow_{w,j} >= 0.20 * D * z_w  (open => serve >= 20%)

Coupling: capital budget K = 150 (thousand $)
  sum_i x_i + sum_w F_w * z_w <= K

Objective: maximize portfolio return - warehouse fixed+variable costs
  (both sides in 'thousand $ of value' units: return in k$, variable ship
   cost 0.10 per unit-km-equivalent, here a flat 0.20 per unit shipped)
"""
from ortools.linear_solver import pywraplp

# ----------------------------- data
r = [0.05, 0.08, 0.12, 0.10, 0.15]
q = [0.5, 1.0, 2.0, 1.5, 3.0]
K = 150.0
SECTOR_CAP = 0.40
MAX_ASSETS = 4
FUND5_TICKET = 3.0
FUND5_CAP = 20.0

F = [40.0, 30.0, 50.0]
C = [60.0, 50.0, 70.0]
d = [40.0, 35.0, 25.0]  # region demands, total 100
D = sum(d)
SHIP_COST = 0.20

s = pywraplp.Solver.CreateSolver("SCIP")
assert s is not None
s.SuppressOutput()

# ----------------------------- portfolio variables
x = [s.NumVar(0, K, f"x_{i+1}") for i in range(5)]
ya = [s.BoolVar(f"y_asset_{i+1}") for i in range(5)]

# ----------------------------- warehouse variables
z = [s.BoolVar(f"z_wh_{w+1}") for w in range(3)]
flow = [[s.NumVar(0, D, f"f_{w+1}_{j+1}") for j in range(3)] for w in range(3)]

# ----------------------------- portfolio constraints
s.Add(sum(q[i] * x[i] for i in range(5)) <= 8.0, "risk_budget")
s.Add(x[2] + x[3] <= SECTOR_CAP * sum(x), "sector_cap_3_4")
s.Add(x[4] >= FUND5_TICKET * ya[4], "fund5_min_ticket")
s.Add(x[4] <= FUND5_CAP * ya[4], "fund5_bigM")
for i in range(5):
    s.Add(x[i] <= K * ya[i], f"asset{i+1}_activation")
s.Add(sum(ya) <= MAX_ASSETS, "at_most_4_assets")
# every activated asset must actually receive money (else the cardinality
# constraint is vacuous: 4 slots burned on zero investments)
for i in range(5):
    s.Add(x[i] >= 0.5 * ya[i], f"asset{i+1}_min_if_active")

# ----------------------------- warehouse constraints
for j in range(3):
    s.Add(sum(flow[w][j] for w in range(3)) == d[j], f"demand_r{j+1}")
for w in range(3):
    s.Add(sum(flow[w]) <= C[w] * z[w], f"cap_wh{w+1}")
    s.Add(sum(flow[w]) >= 0.20 * D * z[w], f"min_share_wh{w+1}")

# ----------------------------- coupling: one shared capital budget
s.Add(sum(x) + sum(F[w] * z[w] for w in range(3)) <= K, "shared_capital")

# ----------------------------- objective
profit = (
    sum(r[i] * x[i] for i in range(5))            # portfolio return
    - sum(F[w] * z[w] for w in range(3))          # warehouse fixed costs
    - SHIP_COST * sum(flow[w][j] for w in range(3) for j in range(3))  # variable
)
s.Maximize(profit)

assert s.Solve() == pywraplp.Solver.OPTIMAL
print(f"Status: OPTIMAL   Objective = {s.Objective().Value():,.3f} (k$)")
print("\nPortfolio (thousand $):")
for i in range(5):
    if x[i].solution_value() > 1e-6 or ya[i].solution_value() > 0.5:
        print(f"  asset {i+1}: invest = {x[i].solution_value():7.2f}  active = {ya[i].solution_value():.0f}")
print("\nWarehouses:")
for w in range(3):
    if z[w].solution_value() > 0.5:
        shipped = sum(flow[w][j].solution_value() for j in range(3))
        print(f"  wh {w+1}: OPEN, throughput = {shipped:6.1f} / capacity {C[w]:.0f}")
    else:
        print(f"  wh {w+1}: closed")
print("\nCapital check: portfolio = {:.2f} + fixed wh costs = {:.2f}  <=  {:.0f}".format(
    sum(x[i].solution_value() for i in range(5)),
    sum(F[w] * z[w].solution_value() for w in range(3)),
    K,
))
