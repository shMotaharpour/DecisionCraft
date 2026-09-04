# Phase 1 — MILP: The Modeling Alphabet (Lesson 1.1)

## 1. What is an optimization model?

Every optimization model has three parts:

1. **Decision variables** — the quantities we get to choose. `x`
2. **Objective function** — a quantity we maximize or minimize, written in
   terms of the variables. `maximize cᵀx`
3. **Constraints** — rules the variables must obey. `Ax ≤ b`

A model with **linear** objective + constraints and **continuous** variables
is a **Linear Program (LP)**. If some variables must be integers (often
0/1 yes/no decisions), it becomes a **Mixed-Integer Linear Program (MILP)**.

## 2. The miniature problem (portfolio, 3 assets)

You have a budget of **$10,000** and three assets:

| Asset | Return per $ | Min investment | Type |
|-------|-------------|----------------|------|
| A     | 0.08        | —              | continuous |
| B     | 0.12        | $2,000 if used | continuous |
| C     | 0.20        | all-or-nothing: invest exactly $5,000 or $0 | binary |

- Choosing B "switches on" a **minimum** investment → needs a binary variable.
- Choosing C is **all-or-nothing** → binary variable itself is the investment.

This single toy problem exercises the three core variable types and two of
the most important modeling tricks (big-M and direct binaries).

## 3. From-scratch model (formulation)

Variables:
- `x_A, x_B ≥ 0` — dollars in A and B (continuous)
- `y_B ∈ {0,1}` — did we "activate" asset B?
- `x_C ∈ {0,1}` — did we invest in C? (binary = the $5,000 amount)

Constraints:
```
x_A + x_B + 5000·x_C ≤ 10000        (budget)
x_B ≤ 10000 · y_B                   (big-M: can only use B if y_B = 1)
x_B ≥ 2000 · y_B                    (if y_B = 1, must invest ≥ $2,000)
x_A, x_B ≥ 0, y_B, x_C ∈ {0, 1}
```

Objective:
```
maximize 0.08·x_A + 0.12·x_B + 0.20·5000·x_C
```

**Why "big-M"?** `x_B ≤ M·y_B` is the standard trick to encode
"if y_B = 0 then x_B = 0". M must be big enough to never bind (here the
budget makes M = 10,000 safe). If M is too small you silently cut off good
solutions; if absurdly big the solver gets numerically slow — always derive
M from a real constraint (here: the budget).

## 4. Run it

```bash
uv run python phase1_milp/lesson1_1_scipy.py         # scipy.optimize.milp
uv run python phase1_milp/lesson1_1_portfolio_ortools.py  # OR-Tools (SCIP)
```

Expected: the solver puts everything it can into the highest-return options —
$5,000 in C (binary on), and the remaining $5,000 in B (activating `y_B`,
which satisfies B's $2,000 minimum). Asset A gets $0 because its 8% return is
dominated by B's 12%. Objective = $1,600. Verify by hand that no feasible
swap beats it.

## 5. Key takeaways

- **Continuous vs integer vs binary** variables encode "how much", "how many",
  and "yes/no".
- **Big-M** converts implications (if-then) into linear constraints.
- MILP is NP-hard in general; solvers (HiGHS, Gurobi, CPLEX) use branch &
  bound + LP relaxations. Small models solve instantly; the skill is in
  *modeling*, not in re-implementing the solver.
- **"This is proven":** linear programming duality and the correctness of
  the simplex method are rigorously proven theorems — you can trust an
  optimal LP answer unconditionally. For MILP, branch & bound convergence
  to proven optimality is also proven (given exact arithmetic and enough
  time).
