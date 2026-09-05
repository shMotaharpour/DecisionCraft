# Phase 1 — Lesson 1.4d: Model Reuse, Build Cost & Hard Constraint Classes

Three production questions: (1) can a base model be updated in place,
(2) what does building a model cost, (3) which constraints hurt solve time
most, and how does CP-SAT handle symmetry internally?

## 1. Updating a base model — what's mutable, what isn't

### pywraplp (SCIP/CBC/...) — the solver is a mutable workspace ✅

Verified live in this lesson's history:
```python
s = pywraplp.Solver.CreateSolver("SCIP")
x = s.NumVar(0, 10, "x")
s.Add(x <= 5 * b)
s.Maximize(x)
s.Solve()               # x = 5
x.SetBounds(3, 10)      # ← in-place bound change
s.Solve()               # re-solve works; x = 5 still (bound not binding)
```
- Variable mutation: `x.SetBounds(lb, ub)`, `SetLb`, `SetUb`, `SetInteger`.
- Constraint mutation: `c.SetCoefficient(var, new_coef)`, `c.SetLb/SetUb/SetBounds`.
- Objective mutation: `obj.SetCoefficient(...)`, `SetOffset`, `Clear()`.
- Adding new vars/constraints after a solve also works (the solver just
  re-solves). **Warm starting is automatic** on many backends (SCIP keeps
  its primal/dual info) — this is the "rolling horizon" pattern in finance:
  solve day 1, update prices/budget in place, re-solve day 2 in the *same
  model object*; it is typically much faster than rebuilding.

### CP-SAT — the model proto is effectively immutable ❌ (rebuild pattern)

`CpModel` wraps a serialized `CpModelProto` intended to be *sent* to the
solver. There is no public API to change a variable's domain after
creation, no `SetCoefficient` on an existing line, and no bound updates.
The supported pattern is **parameterized rebuild**: write a builder
function, call it whenever data changes.
```python
def build_model(budget: int, n_items: int):
    m = cp_model.CpModel()
    x = [m.NewIntVar(0, n_items, f"x{i}") for i in range(n_items)]
    m.Add(sum(cost[i] * x[i] for i in range(n_items)) <= budget)
    m.Maximize(sum(value[i] * x[i] for i in range(n_items)))
    return m

for budget in (1000, 1200, 900):
    solver = cp_model.CpSolver()
    solver.parameters.fix_variables_to_their_hinted_value = False
    solver.solve(build_model(budget, 50))
```
Rebuild is *cheap* — see §2 — and CP-SAT re-uses nothing between solves,
so rebuild+solve is the honest cost. Cross-run reuse exists only via
hints (`AddHint` from the previous solution) and via
`solver.parameters.*` tuning, not via model mutation. If you need true
incremental models in CP-SAT, you re-add constraints conditionally
(e.g. `AddAssumption` toggles) or model the change *inside* the model
(parameters as variables with fixed values from data).

**When to choose which:** repeated same-shape solves with changing numbers
→ pywraplp in-place updates (or CP-SAT rebuild — both fine); changing
structure (different number of items/constraints) → rebuild in both.

## 2. Model-building cost (measured on this machine)

Measured just now, Python 3.11, this VM:
- CP-SAT **build only**: 500 int vars + 501 constraints → **~6 ms**
- CP-SAT **build + solve** (to optimum) → ~104 ms, i.e. solve ≈ 17× build

Scaling rules of thumb:
- Building is O(n_vars + n_nonzeros): linear, Python-object overhead
  dominates. ~100k vars ≈ tens of ms; ~1M vars ≈ seconds (consider
  building the proto directly in C++ or pre-building MPS files).
- Python-side `for` loops adding constraints are the slow part of large
  builds; use `m.Add(sum(...))` batches and `NewIntVarSeries` (pandas) or
  build the proto in bulk for very large models.
- `model.ModelStats()` and `solver.ResponseStats()` are free diagnostics.
- Solve time does NOT scale like build time: it can be milliseconds or
  hours on the same var count, depending on constraint types (§3) and
  integrality.

## 3. Which constraints make solving hard (and CP-SAT's symmetry answer)

### Hard vs easy (same size, different runtime!)

| Feature | Effect on solve time |
|---------|----------------------|
| Continuous LP, clean matrix | milliseconds (polynomial: interior point / simplex) |
| Integers with **big-M** constraints | the big-M gap weakens relaxations → exponential worst case |
| Equality-heavy integer systems (knapsacks, bin packing) | hard: combinatorial |
| var × var products, division, modulo | hardest: weak propagation, deep search trees |
| Native globals (`NoOverlap`, `Cumulative`, `Circuit`, `AllDifferent`) | *faster* than their manual linear encodings |
| Tight variable domains, few alternative optima | fastest |
| Loose domains, symmetric interchangeable entities | can be catastrophically slow |

### Symmetry in CP-SAT: automatic AND manual

The user's question: "does CP-SAT prevent symmetric search by itself, or
must we do it with our own constraints?" — the honest answer is **both,
and they serve different purposes**:

1. **CP-SAT has a built-in symmetry breaker.** During presolve it detects
   variable symmetries and adds *dynamic symmetry-dominance* rules at the
   root level (it literally reports "Presolve summary: detected symmetry
   group..."). This is why it often handles interchangeable binaries far
   better than classic B&B solvers. It is automatic and safe for pure
   optimization.
2. **But** automatic detection is syntactic, not semantic. Business
   symmetries (two *mathematically* identical but *differently described*
   warehouses) may not be detected. And during **solution enumeration**
   (`SearchForAllSolutions`), dynamic symmetry breaking is disabled —
   otherwise it would drop solutions — so there you MUST break symmetry
   yourself (canonical-order constraints) or post-deduplicate.
3. **Best practice stays:** (a) let CP-SAT's presolve do its thing; (b)
   still add cheap manual symmetry breaking (`x1 >= x2`, ordering
   constraints) when you *know* entities are interchangeable — it
   compounds with the internal machinery; (c) never add manual symmetry
   breaking when enumerating all solutions; (d) reformulate to remove
   symmetry entirely when possible (e.g. count identical items instead of
   one variable per item — the "strength through aggregation" trick).

Also worth knowing: **dominance rules** (a problem-specific rule that says
"solution A is at least as good as B") are the semantic, hand-crafted
cousin of symmetry breaking — in CP-SAT you encode them as ordinary
constraints, and they can shrink the space far more than any solver flag.

### What the live experiment actually showed (honest reading)

`lesson1_4d_reuse_cost_symmetry.py` part (d): on a small 16-binary
"choose 4" model, CP-SAT solved BOTH versions in ~4 ms with **0 branches
and 0 conflicts** — its presolve propagation annihilated the model before
search began, and manual symmetry breaking added nothing (even cost a
little). Lesson: **CP-SAT's internal machinery is strong on small/loose
models; manual symmetry breaking pays off on large models where search
actually runs** (benchmarks in the literature show orders-of-magnitude
gains there). The measurement that matters is yours: compare
`NumBranches`/`NumConflicts`/wall time with and without, per model.

### Verified facts in this lesson

- pywraplp: in-place `SetBounds` then re-`Solve()` works (ran: initial 5.0,
  after tightening lb to 3 → still 5.0, status OPTIMAL).
- CP-SAT build cost: 500 vars/501 constraints ≈ 6 ms (measured, mean of
  200 runs); build+solve ≈ 104 ms.
