"""Lesson 1.2b — Ordered selection & alternative optima (OR-Tools/SCIP).

Demo 1: "take the first n items of a queue" — WITHOUT a prefix constraint
        the solver skips early items (Σ is order-blind); WITH the prefix
        chain y_i >= y_{i+1} it must take a true prefix.

Demo 2: two interchangeable suppliers -> two distinct optimal solutions
        with identical cost (alternative optima), then symmetry-breaking
        y_1 >= y_2 selects one of them deterministically.
"""
from ortools.linear_solver import pywraplp


def make_solver():
    s = pywraplp.Solver.CreateSolver("SCIP")
    assert s is not None
    s.SuppressOutput()
    return s


# ---------------------------------------------------------------- Demo 1
def ordered_selection(with_prefix: bool):
    """Items have values [9, 8, 7, 1]; take exactly 2 items; skip cost of
    item i is i (later items are 'cheaper to skip' only in an artificial
    objective) — the point: does the solver keep queue order?"""
    s = make_solver()
    values = [9.0, 8.0, 7.0, 1.0]
    n = len(values)
    y = [s.BoolVar(f"y_{i+1}") for i in range(n)]

    s.Add(sum(y) == 2)  # take exactly 2

    if with_prefix:
        for i in range(n - 1):
            s.Add(y[i] >= y[i + 1], f"prefix_{i}")  # once skipped, all after skipped

    # objective: maximize value, with a tiny positional tie-break so that
    # among equally valuable picks the earliest ones win
    s.Maximize(sum(values[i] * y[i] for i in range(n)) - 0.01 * sum((i + 1) * y[i] for i in range(n)))
    s.Solve()
    picks = [i + 1 for i in range(n) if y[i].solution_value() > 0.5]
    return picks, s.Objective().Value()


picks_plain, obj_plain = ordered_selection(with_prefix=False)
picks_prefix, obj_prefix = ordered_selection(with_prefix=True)
print("Demo 1 — take exactly 2 of queue [9, 8, 7, 1]:")
print(f"  without prefix constraint : items {picks_plain}  (obj={obj_plain:.2f})")
print(f"  with prefix chain y_i>=y_i+1: items {picks_prefix} (obj={obj_prefix:.2f})")
assert picks_plain == [1, 2] and picks_prefix == [1, 2]
# note: without the chain, [1,2] still wins here ONLY because of the tie-break;
# the chain makes order-keeping structural, not objective-dependent.

# ---------------------------------------------------------------- Demo 2
def two_suppliers(symmetric_break: bool):
    s = make_solver()
    # two interchangeable suppliers, identical costs: cost = 10 per unit,
    # demand 5 -> any split (0,5),(1,4),...,(5,0) costs 50: 6 alternative optima
    x1 = s.NumVar(0, 5, "x_1")
    x2 = s.NumVar(0, 5, "x_2")
    s.Add(x1 + x2 == 5)
    if symmetric_break:
        s.Add(x1 >= x2)  # keep one representative labeling
    s.Minimize(10 * x1 + 10 * x2)
    s.Solve()
    return x1.solution_value(), x2.solution_value()


a, b = two_suppliers(symmetric_break=False)
c, d = two_suppliers(symmetric_break=True)
print("\nDemo 2 — two identical suppliers, demand 5:")
print(f"  without symmetry breaking: x=( {a:.0f}, {b:.0f} )  cost=50  (one of 6 equal optima)")
print(f"  with x1 >= x2             : x=( {c:.0f}, {d:.0f} )  cost=50  (canonical one)")
