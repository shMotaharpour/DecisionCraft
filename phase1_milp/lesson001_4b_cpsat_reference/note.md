# Phase 1 — Lesson 1.4b: CP-SAT Complete Constraint Reference

`ortools.sat.python.cp_model` is OR-Tools' second, more modern solver.
Where `pywraplp` (lesson 1.4) wraps classic MILP solvers (SCIP/CBC/GLOP),
**CP-SAT is a SAT-based constraint solver**: it natively handles logic,
scheduling, and integer arithmetic, and is often 10–100× faster than SCIP
on those problem classes. For finance/optimization work you will keep both
in your toolbox: MILP for money-continuous models, CP-SAT for scheduling,
sequencing, and heavy logic.

## CP-SAT vs pywraplp — key structural differences

| Aspect | pywraplp (SCIP) | CP-SAT |
|--------|-----------------|--------|
| Variables | continuous + integer | **integer only** (booleans = 0/1 ints); no floats |
| Linear constraints | native | native (`Add(expr <= k)` also works) |
| Logic (and/or/implication) | manual big-M | **native** — no big-M needed |
| Scheduling | manual | **native** (interval variables, NoOverlap, Cumulative) |
| Objective | linear only | linear; floats must be scaled to ints (multiply by 100 for cents) |
| All-solutions enumeration | no | `solver.parameters.enumerate_all_solutions = True` |
| Status API | codes | `cp_model.OPTIMAL` etc. via `solver.StatusName()` |

Because everything is integer, a variable representing money uses scaled
units (e.g. cents): `price = model.NewIntVar(0, 10**7, "price_cents")`.

## Category map (all commands)

| # | Category |
|---|----------|
| 1 | Variables & constants |
| 2 | Linear & arithmetic constraints |
| 3 | Boolean logic constraints |
| 4 | Combinatorial & table constraints |
| 5 | Routing constraints (circuit) |
| 6 | Scheduling (intervals, NoOverlap, Cumulative) |
| 7 | Objective & hints |
| 8 | Solver parameters, solve & query |

---

## Category 1 — Variables & constants

- `model.NewIntVar(lb, ub, name)` — integer variable; **the only kind**.
- `model.NewBoolVar(name)` — boolean; same as `NewIntVar(0, 1)`.
- `model.NewIntVarFromDomain(cp_model.Domain.FromValues([2,3,5,7]), "p")` —
  variable restricted to a *set* of allowed values (e.g. only tradeable
  lot sizes) — something MILP cannot express in one line.
- `model.NewConstant(v)` — a fixed value usable inside expressions.
- Series helpers (vectorized): `NewIntVarSeries`, `NewBoolVarSeries`
  (pandas-indexed) for bulk variable creation.
- `model.NewIntVar(lb, ub, "")` with `model.SetName` if you need renaming.

## Category 2 — Linear & arithmetic constraints

- `model.Add(expr)` — linear equations/inequalities over ints:
  `model.Add(3*x + 2*y <= 10)`, equalities, chained comparisons.
- `model.AddLinearConstraint(expr, lb, ub)` — explicit-bounds form.
- `model.AddLinearExpressionInDomain(expr, domain)` — linear expression
  restricted to a domain set.
- Non-linear equalities (each *exactly one* relation, not an inequality):
  - `AddMultiplicationEquality(z, [x, y])` — z = x·y (product of two
    variables; linear·constant goes directly in `Add`).
  - `AddDivisionEquality(z, x, y)` — z = x ÷ y (integer division).
  - `AddModuloEquality(z, x, y)` — z = x mod y.
  - `AddAbsEquality(z, x)` — z = |x|.
  - `AddMaxEquality(z, vars)` / `AddMinEquality(z, vars)` — z = max/min.
  - `AddElement(index_var, target_list, target_value)` — target_list[index_var]
    == target_value (lookup by variable index — "choose the j-th warehouse's
    cost" logic).
- Use case note: products/divisions/mods make models much harder — reach
  for them only when linearization is genuinely unnatural.

## Category 3 — Boolean logic constraints (CP-SAT's superpower)

- `AddImplication(a, b)` — a ⇒ b. Direct; no big-M needed.
- `AddBoolOr([a, b, c])` — at least one true.
- `AddBoolAnd([a, b])` — all true.
- `AddBoolXOr([a, b])` — parity.
- `AddAtLeastOne(vars)` / `AddAtMostOne(vars)` / `AddExactlyOne(vars)` —
  cardinality with readability (these replace the `Σy ≤ k` encodings of 1.2;
  k>1 versions via `Add(... )` on sums).
- `AddAllowedAssignments([x, y], [(0,1), (1,0), (2,3)])` — whitelist of
  value tuples (see category 4).
- `AddForbiddenAssignments(...)` — blacklist of tuples.
- Boolean *views*: `x.Not()` — use inside logic constraints.
- `Add(x != y)` works directly for ints (SAT handles disequality natively).

## Category 4 — Combinatorial & table constraints

- `AddAllDifferent(vars)` — all pairwise distinct (e.g. no two trades get
  the same execution slot).
- `AddAllowedAssignments([x, y, z], table)` — the tuple must appear in the
  table (extended "element" logic; encodes any finite input-output map).
- `AddForbiddenAssignments([x, y], table)` — forbid specific combinations.
- `AddInverse(permutation_a, permutation_b)` — b is the inverse permutation
  of a (two-way assignment consistency).
- `AddAutomaton(vars, start, finals, transitions)` — restrict the variable
  sequence to a regular-language pattern (e.g. "max 3 consecutive losing
  days" as a finite-state machine).

## Category 5 — Routing constraints (circuit)

- `AddCircuit(arcs)` — arcs are `(tail, head, literal)` triples; forces a
  single Hamiltonian circuit over selected arcs. The modern way to model
  **TSP / VRP** without MTZ big-M constraints.
- `AddMultipleCircuit(arcs)` — several disjoint circuits (multi-vehicle
  routing: each vehicle = one loop; nodes not visited stay on self-loops).

## Category 6 — Scheduling (interval variables)

The killer feature for workforce/resource scheduling:
- `model.NewIntervalVar(start, duration, end, name)` — a task whose
  `start + duration == end` is enforced automatically.
- `NewOptionalIntervalVar(start, dur, end, presence_literal, name)` — a task
  that may be skipped (if presence = 0, it occupies no resource).
- `NewFixedSizeIntervalVar(start, size, name)` — fixed duration.
- `model.AddNoOverlap(intervals)` — intervals may not overlap (single
  machine, a room, a worker).
- `model.AddNoOverlap2D(intervals_x, intervals_y)` — 2D packing (e.g.
  pallets on shelves).
- `model.AddCumulative(intervals, demands, capacity)` — resource with total
  capacity (workers with skills: several tasks in parallel as long as
  Σ demands ≤ capacity).
- `NewReservoirConstraint` / `WithActive` — level constraint with min/max
  levels over time (cash-flow / inventory level logic!).

## Category 7 — Objective & hints

- `model.Maximize(expr)` / `model.Minimize(expr)` — linear or max/min of
  expressions. CP-SAT objective must be integer-valued.
- `model.ClearObjective()`.
- `model.AddHint(var, value)` / `ClearHints()` — warm start (declare once
  per variable).
- **Assumptions** (`AddAssumption`, `AddAssumptions`, `ClearAssumptions`) —
  temporary constraints you can toggle: solve, and if infeasible ask
  `SufficientAssumptionsForInfeasibility()` *which* assumptions clashed —
  an elegant infeasibility-explainer for decision support.
- `model.AddDecisionStrategy(vars, CHOOSE_FIRST, SELECT_MIN_VALUE)` — steer
  search order (performance tuning only; never changes the answer).

## Category 8 — Solver, parameters & querying

- `solver = cp_model.CpSolver()`; key parameters:
  - `solver.parameters.max_time_in_seconds = 30`
  - `solver.parameters.num_workers = 8` (parallelism — the big speed lever)
  - `solver.parameters.enumerate_all_solutions = True` (with `SearchForAllSolutions`)
  - `solver.parameters.log_search_progress = True`
- `status = solver.Solve(model)` — statuses `OPTIMAL`, `FEASIBLE`,
  `INFEASIBLE`, `MODEL_INVALID`, `UNKNOWN`.
- `solver.SearchForAllSolutions(model, callback)` — enumerate every solution
  (alternative optima, natively! The thing pywraplp struggles with).
- Query: `solver.Value(var)` / `solver.BooleanValue(b)` / `solver.Values(...)`,
  `solver.ObjectiveValue()`, `solver.BestObjectiveBound()`.
- Stats: `solver.StatusName()`, `WallTime()`, `UserTime()`,
  `NumBranches()`, `NumConflicts()`, `NumBooleans()`,
  `num_binary/integer_propagations()`, `SolutionInfo()`,
  `ResponseStats()` (one-line dashboard).
- `model.ExportToFile(path)` / `model.Validate()` / `model.ModelStats()`.

---

## Which solver when? (decision guide)

| Problem trait | Use |
|---------------|-----|
| Continuous money/capacity split, LP-relaxation-friendly | **pywraplp + GLOP/SCIP** (or scipy.optimize.milp) |
| Mostly yes/no & integer decisions, heavy logic implications | **CP-SAT** |
| Scheduling with intervals, machines, calendars | **CP-SAT** (scheduling constraints) |
| Need duals/shadow prices (finance marginal analysis) | **pywraplp LP** (CP-SAT has no duals) |
| Need ALL optimal solutions enumerated | **CP-SAT** (`SearchForAllSolutions`) |

**Why it's proven:** CP-SAT is a SAT solver with integer propagation; its
completeness (proves INFEASIBLE or finds a solution) rests on the
completeness of SAT resolution — a proven result. Optimality claims come
with proven bounds just like B&B (`BestObjectiveBound`).
