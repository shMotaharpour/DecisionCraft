"""Lesson 1.4d — Model mutation vs rebuild, build cost, symmetry handling.

(a) pywraplp: in-place model mutation (bound change + coefficient change)
    then re-solve in the SAME solver object.
(b) CP-SAT: parameterized rebuild builder (no in-place mutation exists).
(c) Build-cost micro-benchmark for CP-SAT.
(d) Symmetry: CP-SAT's internal presolve symmetry detection vs manual
    symmetry breaking — measured on an interchangeable-binaries model.
"""
import time

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model


def banner(t):
    print(f"\n=== {t} ===")


# ------------------------------------------------- (a) pywraplp in-place
banner("(a) pywraplp: mutate in place, re-solve")
s = pywraplp.Solver.CreateSolver("SCIP")
s.SuppressOutput()
x = s.NumVar(0, 10, "x")
b = s.BoolVar("b")
c1 = s.Add(x <= 5 * b, "switch")
s.Maximize(x)
s.Solve()
print(f"  v1: x={x.solution_value():.1f} b={b.solution_value():.0f}")

# update in place: tighter budget, new coefficient, new constraint
x.SetUb(4.0)
c1.SetCoefficient(b, 2.0)  # x <= 2*b now
extra = s.Add(b <= 1, "trivial")
s.Solve()
print(f"  v2 (x<=4, x<=2b): x={x.solution_value():.1f} b={b.solution_value():.0f}")


# ------------------------------------------------- (b) CP-SAT rebuild
def build_budget_model(budget: int, prices, values):
    m = cp_model.CpModel()
    n = len(prices)
    x = [m.NewIntVar(0, 5, f"x{i}") for i in range(n)]
    m.Add(sum(prices[i] * x[i] for i in range(n)) <= budget)
    m.Maximize(sum(values[i] * x[i] for i in range(n)))
    return m


banner("(b) CP-SAT: parameterized rebuild with changing budget")
prices = [30, 45, 20, 60]
values = [40, 50, 25, 90]
for budget in (100, 150, 90):
    model = build_budget_model(budget, prices, values)
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 4
    st = solver.Solve(model)
    obj = solver.ObjectiveValue()
    print(f"  budget={budget:4d}: status={solver.StatusName(st):8s} obj={obj:.0f}")


# ------------------------------------------------- (c) build cost
banner("(c) CP-SAT build-cost micro-benchmark (500 vars, 501 constraints)")
N = 50
t0 = time.perf_counter()
for _ in range(N):
    m = cp_model.CpModel()
    xs = [m.NewIntVar(0, 100, f"x{i}") for i in range(500)]
    m.Add(sum(xs) <= 1000)
    for i in range(500):
        m.Add(xs[i] <= i)
    m.Maximize(sum(xs))
build_ms = (time.perf_counter() - t0) / N * 1000
print(f"  build only: {build_ms:.2f} ms average over {N} runs")


# ------------------------------------------------- (d) symmetry
def symmetric_model(manual_break: bool, n_pairs: int = 8):
    """n_pairs interchangeable item-pairs; choose any 4 items. Without
    breaking, (2^8 * C(8,4)) equivalent labelings exist."""
    m = cp_model.CpModel()
    y = [m.NewBoolVar(f"y{i}") for i in range(2 * n_pairs)]
    m.Add(sum(y) == 4)
    if manual_break:
        # pair i may only be 'opened' if pair i-1 is opened (canonical order)
        for i in range(1, n_pairs):
            m.Add(y[2 * i - 2] >= y[2 * i])
    m.Maximize(sum(y))
    return m


banner("(d) symmetry: identical binaries, choose 4 of 16")
for manual in (False, True):
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 4
    t0 = time.perf_counter()
    st = solver.Solve(symmetric_model(manual))
    dt = time.perf_counter() - t0
    print(f"  manual_break={manual!s:5s}: {solver.StatusName(st):8s} "
          f"branches={solver.NumBranches():7d} conflicts={solver.NumConflicts():6d} "
          f"wall={dt*1000:8.1f} ms")
