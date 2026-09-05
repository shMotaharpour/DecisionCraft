# Phase 1 — Lesson 1.4c: Callbacks in CP-SAT & Constraint Cost / Best Practices

Two follow-ups from lesson 1.4b: (1) the callback object model, (2) the
computational cost of different constraints and how to build fast models.

## Part A — The callback object model (`CpSolverSolutionCallback`)

### What the object is

`CpSolverSolutionCallback` is a **C++ observer** that the solver calls back
into Python every time it finds a new solution. You subclass it and override
**`on_solution_callback(self)`** — the single required hook. The solver is
C++ (fast); Python code runs only when a solution is found, so a callback is
the sanctioned escape hatch for streaming results out of the search.

### The base class API (what a subclass gets for free)

| Method | Meaning |
|--------|---------|
| `self.Value(expr)` / `self.value(expr)` | evaluate a variable or linear expression in the *current* solution |
| `self.BooleanValue(lit)` | value of a boolean var or its negation `lit.Not()` |
| `self.FloatValue(expr)` | float view (still computed from ints) |
| `self.ObjectiveValue()` / `self.BestObjectiveBound()` | current incumbent / proven bound |
| `self.NumConflicts()`, `self.NumBranches()`, `self.NumBooleans()`, `self.NumBinaryPropagations()`, `self.NumIntegerPropagations()` | live search stats |
| `self.WallTime()`, `self.UserTime()`, `self.DeterministicTime()` | clocks |
| `self.Response()` / `self.ResponseProto()` | the whole raw response proto |
| `self.StopSearch()` | **asynchronously abort the search** from inside the callback |
| `self.HasResponse()` | whether a solution exists yet |

⚠️ Inside a callback, always use `self.Value(...)` — the outer solver's
`Value()` is not yet populated mid-search.

### Correct definition pattern

```python
class Collector(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        super().__init__()          # REQUIRED, and no args (the base takes none)
        self.variables = variables  # keep var references; do NOT copy values here
        self.seen = []              # your own state

    def on_solution_callback(self):
        # read values ONLY via self.* helpers
        self.seen.append(tuple(self.Value(v) for v in self.variables))
        # optional early stop:
        if len(self.seen) >= 100:
            self.StopSearch()
```

Rules:
1. Call `super().__init__()` first — the SWIG proxy must be wired.
2. Never read variable values from outside the callback mid-search; only
   `self.Value`/`self.BooleanValue` reflect the *current* solution.
3. `on_solution_callback` runs once per **improving/feasible** solution found
   — not per node, not per restart.
4. To enumerate all solutions you must set
   `solver.parameters.enumerate_all_solutions = True` (otherwise the solver
   stops after the optimum and your callback fires once or twice).

### When callbacks fire — three usage patterns

| Pattern | Setup | Typical use |
|---------|-------|-------------|
| Enumerate all solutions | `SearchForAllSolutions(model, cb)` + `enumerate_all_solutions=True` | alternative optima, counting, constraint sampling |
| Stream improving solutions during optimization | `solver.Solve(model, cb)` with an objective | progress dashboards, "good enough, stop now" via `StopSearch` |
| Feasibility observer (no objective) | same, model without objective | solution pools, enumeration of feasible configs |

Related classes you may subclass:
- **`cp_model.CpSolverSolutionCallback`** — the only Python callback class
  in `ortools.sat.python.cp_model` (verified: nothing else public).
- C++ has richer hooks (`NewFeasibleSolutionObserver`); from Python the
  single-callback design is intentional.
- For logging rather than per-solution work, prefer `solver.parameters.log_search_progress = True` or `solver.log_callback = print` (fires on log lines, not solutions) — far cheaper than a Python callback.

### Cost warning

Each `on_solution_callback` is a Python↔C++ boundary crossing with proto
conversion. In a model that finds millions of solutions, a heavy Python
callback becomes THE bottleneck (can be 100× slower than pure search).
Best practice: inside the callback, extract the minimum (or aggregate),
and post-process afterwards.

## Constraint computational cost & best practices

### Cost hierarchy (rough, CP-SAT-specific)

| Constraint | Relative cost | Why |
|-----------|--------------|-----|
| `Add(linear ==/<= k)` | ★ free | native propagation |
| `AddImplication` / `AddBoolOr/And` | ★ free | SAT clauses — this is the solver's native language |
| `AddAtLeastOne/AtMostOne/ExactlyOne` | ★ free | special-cased cardinality |
| `AddAllDifferent` | ★★ cheap but global | pairwise propagation, stronger than n² inequalities |
| `AddAllowedAssignments` (table) | ★★★ medium | decomposition into many clauses; small tables fine, huge tables slow |
| `AddMultiplicationEquality(z, [x, y])` (var×var) | ★★★★ expensive | introduces an auxiliary booleans decomposition per product bit |
| `AddDivisionEquality`, `AddModuloEquality` | ★★★★ expensive | search-based, weak propagation |
| `AddAutomaton`, `AddCircuit` | ★★–★★★ | global, usually *cheaper than their manual encodings* |
| `AddCumulative`, `AddNoOverlap` | ★★ | global scheduling propagators, well-optimized |

### Best practices (with the toy example below)

1. **Stay linear as long as possible.** Every var×var product costs orders
   of magnitude more than a linear constraint. If one side is a constant,
   `m.Add(2 * x <= 10)` is linear; only `x*y` needs the expensive path.
2. **Scale money to integers.** CP-SAT has no floats: work in cents or
   basis points. (`price_cents = model.NewIntVar(0, 10**7, ...)`)
3. **Prefer native globals over manual encodings.** `AddCircuit` beats a
   hand-written MTZ formulation; `AddNoOverlap` beats pairwise
   `end_a <= start_b` chains; `AddExactlyOne` beats `Σ == 1` (same math,
   better propagation paths and cleaner models).
4. **Tighten variable domains.** `NewIntVar(9, 17, ...)` prunes search
   before propagation even starts; `NewIntVar(0, 10**9, ...)` wastes it.
   Domain tightness is the #1 lever in CP-SAT.
5. **Use symmetries deliberately** (lesson 1.2b): with
   `SearchForAllSolutions` symmetry-breaking is forbidden (removes
   solutions); during pure optimization it is essential.
6. **Tune `num_workers` first** (default uses all cores; matching your
   machine avoids oversubscription in services), then `max_time_in_seconds`.
7. **Warm-start with `AddHint`** when you have yesterday's solution —
   halves time-to-first-solution on many operational models.
8. **Profile, don't guess:** `solver.ResponseStats()` + compare
   `NumConflicts`/`NumBranches` between formulations of the same model.

### Miniature demo

`phase1_milp/lesson1_4c_callbacks_cost.py` measures: (a) a Collector
callback aggregating all solutions of a small model, (b) an early-stopping
callback, (c) the same counting model with a var×var product vs pure linear
— showing the objective/bound gap and runtime difference.