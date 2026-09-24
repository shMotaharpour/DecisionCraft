# Phase 1 — Lesson 1.4: Complete OR-Tools pywraplp Reference (Category 0)

Category 0 lists ALL command categories with their numbering; categories 1–7
then document every command in each category.

Everything below is verified against the installed OR-Tools version
(`from ortools.linear_solver import pywraplp`).

---

## Category 0 — Command map

| # | Category | Purpose |
|---|----------|---------|
| 1 | Solver creation & backends | instantiate the solver object |
| 2 | Variables | define decision variables |
| 3 | Constraints | add linear constraints |
| 4 | Objective | set goal & offset |
| 5 | Solving & control | run the solve, limits, interruption |
| 6 | Solution querying | read results after solve |
| 7 | I/O, proto & diagnostics | export/import models, stats |

---

## Category 1 — Solver creation & backends

- `pywraplp.Solver.CreateSolver(name)` → the entry point. `name` picks the
  backend: `"GLOP"` (pure LP), `"SCIP"` (MILP, open-source), `"SAT"` /
  `"BOP"` (integer/boolean special cases), `"CBC"`, `"CLP"`, `"GLPK"`,
  `"CPLEX"`, `"GUROBI"`, `"XPRESS"`, `"PDLP"` (first-order LP).
  Returns `None` if that backend isn't compiled in — always assert.
- Constants to check capability: `SolverVersion()`, `IsMip()`,
  `SupportsProblemType(...)`.
- Backend constants (e.g. `SCIP_MIXED_INTEGER_PROGRAMMING`,
  `GLOP_LINEAR_PROGRAMMING`) are alternative enum names for the same strings.

## Category 2 — Variables

- `solver.NumVar(lb, ub, name)` — continuous variable.
- `solver.IntVar(lb, ub, name)` — general integer.
- `solver.BoolVar(name)` — binary; same as `IntVar(0, 1)`.
- `solver.Var(lb, ub, integer, name)` — full form (integer flag).
- `solver.NumVariables()`, `solver.variables()`, `solver.LookupVariable(name)`.
- On a Variable object:
  - read: `solution_value()`, `lb`, `ub`, `name`, `index`, `integer`,
    `reduced_cost` (LP duality), `basis_status`.
  - write: `SetBounds(lb, ub)`, `SetLb`, `SetUb`, `SetInteger(bool)`,
    `SetBranchingPriority(n)` (hint which variables to branch first —
    useful when integer variables are "hard").

## Category 3 — Constraints

- `solver.Constraint(lb, ub)` → returns a `Constraint` row you then fill:
  `c.SetExpression` is implicit via `c.SetCoefficient(var, coef)`.
  `c.SetLb / SetUb / SetBounds(lb, ub)` adjust the row later.
- `solver.Add(expr)` — the natural-syntax wrapper used in all our lessons:
  `s.Add(x1 + 2*x2 <= 10, "name")`. Supports `<=`, `>=`, `==`.
- `solver.RowConstraint(lb, ub)` — alias of Constraint.
- `solver.Sum(vars)` — fast linear sum helper.
- On a Constraint object: `GetCoefficient(var)`, `SetCoefficient`,
  `DualValue` (shadow price — LP only, very important in finance:
  marginal value of relaxing the constraint), `lb`, `ub`, `name`, `index`,
  `set_is_lazy(True)` (lazy constraint: only enforced when violated —
  advanced, solver checks it on demand).
- `solver.NumConstraints()`, `solver.constraints()`, `solver.LookupConstraint(name)`,
  `solver.ComputeConstraintActivities()` (debugging: actual row activity values).

## Category 4 — Objective

- `solver.Maximize(expr)` / `solver.Minimize(expr)` — the one-call form.
- `solver.Objective()` → the Objective object for incremental building:
  `obj.SetCoefficient(var, coef)`, `obj.GetCoefficient(var)`,
  `obj.SetOffset(const)` (a constant added to the objective, doesn't affect
  the optimal solution but useful for cost accounting),
  `obj.SetOptimizationDirection(...)` / `SetMaximization` / `SetMinimization`,
  `obj.Clear()`.
- Read after solve: `obj.Value()`, `obj.BestBound()` (the relaxation bound —
  compare with Value to compute the MIP gap yourself:
  `gap = (bound - value)/abs(value)`).

## Category 5 — Solving & control

- `solver.Solve()` → returns status constant (see category 6). The only call
  that actually optimizes.
- `solver.SetTimeLimit(ms)` — wall-clock limit in milliseconds; a must in
  production (a model that runs 10 minutes "until optimal" is useless).
- `solver.SetNumThreads(n)` — parallel B&B.
- `solver.SetSolverSpecificParametersAsString("...")` — raw backend options
  (e.g. SCIP settings).
- `solver.SetHint(variables, values)` — warm start: give a good initial
  solution (from a heuristic or yesterday's answer) to speed up B&B.
- `solver.InterruptSolve()` — stop from another thread, keeping the best
  solution found so far.
- `solver.EnableOutput()` / `solver.SuppressOutput()` — solver log on/off.
  (Warning: names are inconsistent across versions — 1.15 has SuppressOutput.)

## Category 6 — Solution querying

- Status constants: `OPTIMAL`, `FEASIBLE` (time/iteration limit hit but a
  valid solution exists), `INFEASIBLE` (constraints contradict), `UNBOUNDED`
  (objective can go to infinity — check you didn't forget a constraint),
  `ABNORMAL`, `NOT_SOLVED`, `MODEL_INVALID`.
  ⚠️ Always compare against `solver.Solve()`'s return; a solution is only
  trustworthy when status ∈ {OPTIMAL, FEASIBLE}.
- `solver.Objective().Value()` / `.BestBound()`.
- `var.solution_value()` for each variable.
- `solver.NextSolution()` — for some backends, iterate over alternative
  optima (rarely supported; don't rely on it).
- `solver.VerifySolution(is_optimal_only, log_errors)` — feasibility check.
- `solver.WallTime()` (ms), `solver.Iterations()` (simplex/B&B iterations),
  `solver.nodes()` (B&B nodes).

## Category 7 — I/O, proto & diagnostics

- `solver.ExportModelAsLpFormat(False)` / `ExportModelAsMpsFormat(False, False)`
  → string of the model in LP/MPS format. **Best debugging tool when a model
  behaves unexpectedly: read the exported file and check the math is what
  you *meant*.**
- `solver.WriteModelToMpsFile(path)`.
- Proto path: `ExportModelToProto` / `LoadModelFromProto(...)` /
  `LoadSolutionFromProto` / `FillSolutionResponseProto` — build once, ship to
  a service that re-solves; also used with `SolveWithProto`.
- `solver.ComputeConstraintActivities()` — for debugging infeasibility:
  shows each row's actual activity; the row that can't meet its bounds is
  your conflict.
- `Infinity()` — portable "no bound" constant.

---

## Cheat sheet — the 20% you'll use daily

```python
solver = pywraplp.Solver.CreateSolver("SCIP")          # 1
x = solver.NumVar(0, 100, "x"); b = solver.BoolVar("b") # 2
solver.Add(3*x + 5 <= 2*b + 10, "c1")                   # 3
solver.Maximize(2*x - b)                                # 4
solver.SetTimeLimit(30000)                              # 5
status = solver.Solve()                                 # 5
assert status == pywraplp.Solver.OPTIMAL                # 6
print(x.solution_value(), solver.Objective().Value())   # 6
open("model.lp","w").write(solver.ExportModelAsLpFormat(False))  # 7
```

## Mapping to scipy.optimize.milp

| OR-Tools | scipy.optimize.milp |
|----------|---------------------|
| `NumVar/IntVar/BoolVar` | columns of the matrix + `integrality` + `Bounds` |
| `solver.Add(Ax ≤ b)` | `LinearConstraint(A, lb, ub)` rows |
| `Maximize/Minimize` | sign of `c` (scipy always minimizes) |
| `SetTimeLimit(ms)` | `options={"time_limit": seconds}` |
| `SetHint` | `milp(..., x0=...)` |
| status OPTIMAL/INFEASIBLE | `res.status` 0/2 (see `res.message`) |
| `Objective().Value()` | `-res.fun` for maximization |

**Why it's proven:** all backends implement the same math (simplex/duality
for LP, branch & bound for MILP); the wrapper only translates your natural
syntax into matrix form. Choosing GLOP vs SCIP changes only *speed and
capabilities* (SCIP handles integers), never correctness of the two
algorithms' outputs.