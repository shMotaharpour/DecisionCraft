"""Lesson 4.1 — Game theory foundations, solved with LPs (phase-1 toolbox).

(a) best-response analysis of a 2x2 general-sum market-entry game
(b) pure Nash equilibria enumeration
(c) zero-sum variant solved as an LP: minimax = maximin (von Neumann),
    mixed strategies from LP duality (OR-Tools/SCIP)
"""
import itertools

import numpy as np
from ortools.linear_solver import pywraplp

# ------------------------------------------------------------- the game
# Payoffs (row, col). Market entry: {Low, High} capacity.
# General-sum: crowding hurts both, but entering High alone is lucrative.
G = np.array([
    [(6, 6), (3, 8)],    # row=Low:  vs Low (6,6), vs High (3,8)
    [(8, 3), (1, 1)],    # row=High: vs Low (8,3), vs High (1,1)
], dtype=object)
ACTIONS = ["Low", "High"]


def best_responses(G):
    """best_responses[i] = set of opponent actions where action i is a BR."""
    br = {0: [], 1: []}
    for col in range(2):
        vals = [G[r, col][0] for r in range(2)]
        for r in range(2):
            if vals[r] == max(vals):
                br[0].append((r, col))
    for row in range(2):
        vals = [G[row, c][1] for c in range(2)]
        for c in range(2):
            if vals[c] == max(vals):
                br[1].append((row, c))
    return br


br = best_responses(G)
print("(a) best-response analysis (general-sum market entry)")
for (r, c) in itertools.product(range(2), repeat=2):
    is_ne = (r, c) in br[0] and (r, c) in br[1]
    tag = "  <-- NASH" if is_ne else ""
    print(f"    profile ({ACTIONS[r]:4s},{ACTIONS[c]:4s}) payoffs "
          f"({G[r,c][0]},{G[r,c][1]}){tag}")

# ------------------------------------------------------------- zero-sum LP
print("\n(b) zero-sum variant: M = row payoff only")
M = np.array([[m[0] for m in row] for row in G], dtype=float)
print(M)

solver = pywraplp.Solver.CreateSolver("SCIP")
solver.SuppressOutput()
v = solver.NumVar(-solver.infinity(), solver.infinity(), "v")
p = [solver.NumVar(0, 1, f"p{j}") for j in range(2)]
solver.Add(sum(p) == 1)
# every column must earn at most v against p:  Σ_i M[i,j] p_i >= v
for j in range(2):
    solver.Add(M[0, j] * p[0] + M[1, j] * p[1] >= v)
solver.Maximize(v)
assert solver.Solve() == pywraplp.Solver.OPTIMAL
p_star = np.array([p[i].solution_value() for i in range(2)])
value = solver.Objective().Value()
print(f"row player's maximin (LP): value = {value:.4f}, p* = {np.round(p_star, 4)}")

# column player's LP (minimax)
solver2 = pywraplp.Solver.CreateSolver("SCIP")
solver2.SuppressOutput()
w = solver2.NumVar(-solver2.infinity(), solver2.infinity(), "w")
q = [solver2.NumVar(0, 1, f"q{j}") for j in range(2)]
solver2.Add(sum(q) == 1)
for i in range(2):
    solver2.Add(M[i, 0] * q[0] + M[i, 1] * q[1] <= w)
solver2.Minimize(w)
assert solver2.Solve() == pywraplp.Solver.OPTIMAL
q_star = np.array([q[i].solution_value() for i in range(2)])
print(f"col player's minimax (LP): value = {solver2.Objective().Value():.4f}, "
      f"q* = {np.round(q_star, 4)}")

print("\n(c) von Neumann check: maximin == minimax?",
      abs(value - solver2.Objective().Value()) < 1e-9,
      f"(|diff| = {abs(value - solver2.Objective().Value()):.2e})")
print("expected payoff of (p*, q*):", (M @ q_star) @ p_star)
