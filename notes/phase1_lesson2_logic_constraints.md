# Phase 1 — Lesson 1.2: Logic → Linear Constraints

The core *modeling* skill of MILP: translating business rules ("if X then Y",
"choose at most 3", "pay a fixed cost only if used") into linear constraints
over binary variables.

## 1. The toolbox

Let `y ∈ {0,1}` be a binary and `M` a known upper bound on some quantity.

| Rule | Encoding |
|------|----------|
| If `y=1` then `x ≤ U` | `x ≤ U·y` |
| If `y=1` then `x ≥ L` | `x ≥ L·y` |
| If `y=0` then `x = 0` | same as first rule |
| Exactly one of `y₁..yₙ` | `Σyᵢ = 1` |
| At most k of `y₁..yₙ` | `Σyᵢ ≤ k` |
| At least k of `y₁..yₙ` | `Σyᵢ ≥ k` |
| `y` implies `z` | `y ≤ z` |
| If `x > 0` then pay fixed cost `F` | `x ≤ U·y` and add `F·y` to the objective |

**Why it's proven:** any "reasonable" Boolean condition can be written as a
linear inequality over binaries — this is the basis of integer programming
(Balas, extended formulation theory). You never need to memorize proofs;
the table above covers 95% of practical cases.

## 2. The miniature problem — warehouse fixed-cost + picking 2 of 4

You run a warehouse that buys one product type from **4 suppliers**:

- Supplier i charges price `pᵢ` per unit and a **fixed order fee** `Fᵢ`
  (pay the fee only if you order > 0 units from them).
- You may use **at most 2 suppliers** (contract limit).
- You need exactly **100 units**.
- Each supplier has capacity `Uᵢ`.

Data: p = [10, 12, 9, 11], F = [100, 80, 150, 60], U = [60, 50, 70, 40].

Model:
```
minimize  Σ pᵢ·xᵢ + Σ Fᵢ·yᵢ
s.t.      Σ xᵢ = 100                     (demand)
          xᵢ ≤ Uᵢ·yᵢ   ∀i               (fixed-cost activation + capacity)
          Σ yᵢ ≤ 2                       (at most 2 suppliers)
          xᵢ ≥ 0, yᵢ binary
```

Note how ONE constraint pair (`xᵢ ≤ Uᵢ·yᵢ` + `Fᵢ` in the objective) encodes
the entire fixed-cost logic: order 0 → pay nothing; order anything → fee
must be paid.

## 3. Run it

```bash
uv run python phase1_milp/lesson1_2_logic_scipy.py
uv run python phase1_milp/lesson1_2_logic_ortools.py
```

**Think first (answer before looking at output):** the cheapest unit price is
supplier 3 (p=9) but its fee is the largest (150) and you need 100 units but
supplier 3 caps at 70 — so you *must* use a second supplier. Which pair
minimizes total cost?

## 4. Key takeaways

- Fixed costs = one binary + one big-M constraint + one term in the objective.
- Cardinality rules ("at most k") are a single linear inequality over binaries.
- The LP relaxation (dropping integrality) gives a *bound*, not the answer:
  with continuous y, the solver could "half-pay" fees. Integer variables make
  the model business-true — this is exactly why we need MILP instead of LP.