# Phase 2 — Lesson 2.8: Design Tradeoffs I — Resolution (Time & State)

Before any algorithm runs, a modeler decides **how finely to see the
world** — how often decisions happen (time granularity) and how
finely states are distinguished (state granularity). Both are genuine
tradeoffs with the same shape: **more resolution buys control/value and
costs |S|·|A| (and money)**. This lesson measures both dials on live
models; lessons 2.9 (reward & terminal design) and 2.10 (constraints &
deployment) continue the design-tradeoff trilogy.

Evidence: `docs/research/phase2_lesson8a_time_granularity_evidence.txt`
and `...8b_state_granularity_evidence.txt` (live runs).
Demos: `phase2_mdp/lesson2_8a_time_granularity.py`,
`phase2_mdp/lesson2_8b_state_granularity.py`.

## 1. Time granularity — one season, three clocks

Same markdown problem (42-day season, fad-decaying price-sensitive
Poisson demand, price menu {7, 10, 13}): the planner picks a price per
epoch; demand accrues daily inside it. Three clocks: weekly (6 epochs),
2-day (12), daily (42). Exact backward induction each:

| clock | epochs | \|S\| | V(s=15) | solve |
|---|---|---|---|---|
| weekly | 6 | 147 | 145.78 | 25 ms |
| 2-day | 12 | 273 | 147.11 | 51 ms |
| daily | 42 | 903 | 147.31 | 183 ms |

Verified live: value is non-decreasing in fineness (weekly 145.78 ≤
2-day 147.11 ≤ daily 147.31) — a finer clock strictly enlarges the
policy class, so it can only help *the model*. What it costs: |S|
grows linearly in epochs, and every epoch is a decision you must
commit to. The weekly plan cannot reprice after a mid-week demand
shock; the daily plan pays for that repricing option whether or not it
ever fires.

**Methodological trap found during the build (kept in the evidence):**
the first version charged holding cost on the *end-of-epoch leftover*,
which undercharged coarse clocks (stock waits *during* the epoch —
leftover after a week is nearly zero). Fix: hold on average inventory
`(s + leftover)/2 × days_per`, same convention at every clock. With a
shared accounting convention the monotonicity the theory promises
actually appears. **A granularity comparison is only as honest as the
cost accounting it shares.**

Rule of thumb: refine the clock until the marginal value of the last
refinement stops paying for its |S|/|A| cost. When actions have
meaningful duration (not price tags but campaigns, hires, builds), the
right machinery is the **semi-MDP** (SMDP): decisions at epoch
boundaries, actions spanning many primitives — the same lens under
which Kaggriculture's 24-step days (2.3's episodic trick) become one
decision point.

## 2. State granularity — the resolution dial on the state axis

Same inventory MDP; three resolutions of the stock axis (exact 21
levels; 7 levels step-3; 3 levels). The coarse models solve faster —
but their value functions are *self*-estimates. The honest number is
the policy **lifted back** to the true state space and rolled out:

| resolution | \|S\| | solve | true perf |
|---|---|---|---|
| exact | 21 | 41 ms | 386.71 |
| bin(3) | 7 | 14 ms | 384.63 (−0.54%) |
| bin(10) | 3 | 6 ms | 377.55 (−2.37%) |

Reads: bin(3) is nearly free (−0.5% for 3× fewer states); bin(10)
visibly hurts because the (s,S) reorder structure needs resolution
where the value function bends. The coarse model's own V is optimistic
garbage (projection error) — never report it as performance; evaluate
lifted.

Rule of thumb: **bin where the value function is flat; keep resolution
where it bends** — the direct operational form of lesson 2.4's curse
and its ALP/approximation answers.

## 3. The shared shape (why these are the same tradeoff)

Both dials trade *expressiveness* against *model size*: finer time and
finer state each enlarge the policy class (value can only improve in
the model's own units) while multiplying computation — and both have a
plateau where marginal refinement stops paying. The art is spending
resolution only where the solution structure actually bends: clocks at
the moments decisions matter, bins where value is flat.

## 4. Bridge

- Upstream: 2.3's Axis 3 (horizon) and Axis 1 (state type) name the
  dials; here they are measured. 2.4's curse is the cost curve of the
  state dial.
- Downstream: 2.9 continues with reward & terminal-value design (the
  *what are we optimizing* dials); 2.10 closes with constraint-vs-
  penalty and precompute-vs-recompute (the *how it runs* dials).
