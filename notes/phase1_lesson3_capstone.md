# Phase 1 — Lesson 1.3 (Capstone): Portfolio + Warehouse Allocation

The phase-1 project combines everything from 1.1–1.2 into one realistic
model, solved with OR-Tools/SCIP: `phase1_milp/lesson1_3_capstone.py`.

## Business setting

An investment & logistics company must decide, for the coming quarter:

1. **Portfolio side** — allocate capital across 5 assets:
   - each asset has an expected return and a risk exposure;
   - assets 3 and 4 belong to the same sector → **sector cap**: at most 40%
     of the *sector group*;
   - asset 5 is an alternative fund: **all-or-nothing** minimum ticket of
     $3,000 (binary activation);
   - total **risk budget**: Σ (risk_i · x_i) ≤ 8.0;
   - **cardinality**: invest in at most 4 assets (operational limit).

2. **Warehouse side** — one product, three candidate warehouses with
   fixed opening costs and capacities; demand of 3 regions must be served;
   if a warehouse opens it must serve ≥ 20% of total demand (big-M +
   minimum-throughput logic).

3. **The coupling (what makes it a true capstone):** total capital is one
   budget shared by BOTH sides — every dollar invested in the portfolio is
   a dollar not spent opening warehouses. The single MILP balances them.

## Modeling techniques exercised

- binary activation + minimum ticket (1.1)
- fixed costs via objective terms (1.2)
- cardinality constraint (1.2)
- sector cap as a group constraint (1.2 table, "at most k" variant)
- minimum-throughput-if-open: `Σ_i flow_{w,i} ≥ 0.2·D_total · y_w` (1.2)
- assignment structure: warehouse→region flows (1.2b slot pattern's cousin)

## Run

```bash
uv run python phase1_milp/lesson1_3_capstone.py
```

## Key takeaways

- Real models are *composition*: each business rule maps to one small
  linear piece; the skill is keeping the pieces orthogonal and the
  big-Ms tight.
- A shared-resource constraint (here: one capital budget) is what turns
  two unrelated subproblems into one meaningful optimization.
- **Reality check on infeasibility:** our first data set made the model
  INFEASIBLE — no single warehouse can cover demand 100, and the two
  cheapest openings (120+90) exceeded the whole capital K=150. The solver
  is not "broken"; it is telling you the *business requirements contradict
  each other*. Debug order: relax the tightest constraint, re-solve, and
  identify which rule bites.
- **Vacuous activation:** without a "min investment if active" constraint
  (`x_i >= 0.5 * ya_i`), the solver burned 3 of the 4 asset slots on
  y=1 with x=0 (harmless to the objective but wrong business-wise).
  Every activation binary deserves BOTH an upper (big-M) and a lower
  (minimum ticket) tie to its continuous variable.