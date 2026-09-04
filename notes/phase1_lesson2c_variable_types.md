# Phase 1 — Lesson 1.2c: Variable Types & Parameters vs Variables

Two clarifications that came up in discussion after lesson 1.2b.

## 1. Discrete variables are NOT only 0/1

| Type | Domain | Typical use |
|------|--------|-------------|
| **Continuous** | any real number | money, product units |
| **Binary** | {0, 1} | yes/no, on/off |
| **General Integer** | any integer in a range, e.g. {0..10} | machine counts, shifts, factory capacity |

Example — "how many overtime shifts today?" is an integer decision, not a
yes/no:

```
m ∈ {0, 1, 2, 3}        integer variable, NOT binary
m ≤ 3·y                 y binary: overtime allowed at all (yes/no)
```

Binary is just the special case `lb=0, ub=1`. All the logic tricks
(big-M, activation) apply to general integers too.

Code support:
- **scipy** (`scipy.optimize.milp`): `integrality=0` continuous, `integrality=1`
  integer; binaries are integers with bounds `[0, 1]`.
- **OR-Tools** (`pywraplp`): `NumVar` continuous, `IntVar(lb, ub)` integer,
  `BoolVar` binary.

## 2. Parameters vs decision variables — "why wasn't n=5 in the constraints?"

- **Parameter:** a number known in advance (demand=100, prices, budget).
  It enters constraints as a *constant*.
- **Variable:** what the solver decides. The solver is free to pick its value.

In lesson 1.2b we wrote `Σ yᵢ == 2` — the count WAS in the constraint, but as
a *parameter* (fixed 2), not a decision. Three patterns:

1. **n given in advance** (parameter): `Σ yᵢ = n` with n a constant.
2. **n itself part of the decision** (let the economics decide):
   ```
   maximize  Σ vᵢ·yᵢ
   s.t.      Σ cᵢ·yᵢ ≤ B       # budget implicitly decides how many
   ```
   No `Σ = n` constraint at all — the count emerges.
3. **n variable but bounded:** `Σ yᵢ ≤ n_max`.

If the count itself must be an explicit integer decision, use an integer
variable `m` with `Σ yᵢ = m`, `m ∈ {1..n_max}` — but pattern 2 is usually
cleaner (fewer variables).

**Rule of thumb:** anything that must be *decided* is a variable; anything
that is *data* is a parameter. Moving a number between the two roles only
changes where it appears (objective/constraints vs input data), not the
structure of the model.