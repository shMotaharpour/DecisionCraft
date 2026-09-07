# Phase 2 — Lesson 2.5: Absorbing Chains, Hitting Times & Fitting P from Data

Evidence: `docs/research/phase2_lesson5_absorbing_evidence.txt` (live run).
Demo: `phase2_mdp/lesson2_5_absorbing_fit.py`.

Lesson 2.1 answered "where does the chain END UP?" (π\*). This lesson answers
the twin questions "HOW LONG until …?" and "WHICH end?" — the mathematics of
ruin, default, sell-out, and time-to-event. Plus: where P comes from when you
have data instead of a textbook.

## 1. Absorbing chains — canonical form and the fundamental matrix

Order states as (absorbing first | transient). Then P in block form:

```
P = [ I  0 ]
    [ R  Q ]        Q: transient→transient, R: transient→absorbing
```

The **fundamental matrix** `N = (I − Q)⁻¹` (proven invertible for any
absorbing chain: spectral radius of Q < 1) concentrates everything:

| Quantity | Formula | Reading |
|---|---|---|
| E[steps before absorption] from i | `t = N·1` | row i of N sums |
| P(absorb in absorbing state j) | `B = N·R` | row i of B |
| Var[steps] | `t_var = (2N − I)·t − t∘t` | for confidence bands |

Workhorse example (gambler's ruin, stake 0..6, p=0.45 — live): expected
steps to ruin-or-target by stake `[4.29 7.3 8.77 8.33 5.58]`, ruin
probability from fundamental matrix matches the closed form
`(ρ^i − ρ^N)/(1 − ρ^N)` with ρ = q/p to 1.1e-16. Note the asymmetry: the
expected **time** peaks mid-range (8.77) while ruin probability falls
monotonically — two different questions, two different rows of the same N.

Finance mapping: stake = remaining capital, ruin = stop-loss hit, target =
take-profit; B[:, 0] is the "blow-up probability curve" of a doubling-down
strategy. Same math for time-to-default, time-to-sell-out, time-to-hire.

## 2. Mean first passage times without absorbing states

For a recurrent chain, `m_ij` = expected steps to reach j from i obeys the
first-step relation (proven by conditioning on the first jump):

```
m_ij = 1 + Σ_{k ≠ j} P_ik · m_kj        (a linear system per target j)
```

Implement as one `np.linalg.solve(I − P[−j,−j], 1)` per column. Live results
for lesson 2.1's market chain (verified vs Monte Carlo, 60k runs each:
15.96 vs 16.00): bull→stagnant 16 months, bear→bull 4.67, stagnant→bull 4.

**Kemeny's law** (proven): mean return time to i (leave, then come back)
= `1/π_i` — the stationary distribution of lesson 2.1 is exactly a table of
recurrence times: π_bull = 0.60 means "bull returns every 1.67 months".
The two lessons are one theory.

Practical reading: hitting times are the "expected time between regimes" —
the horizon over which a regime-based trading policy must re-forecast.

## 3. Fitting P from data (the step 2.1 skipped)

Textbook P came from nowhere; real pipelines estimate it:
`P̂_ij = n_ij / Σ_k n_ik` — maximum likelihood, just row-normalized counts.

Verified live on 20k days generated from the known 3-regime P:
max |P̂ − P_true| = 0.0099 — clean when every state is visited (the ML
estimator is unbiased for fixed horizon, consistent always). Two real traps:

1. **Zero-visit rows** make P̂ undefined *silently* (0/0 — row of NaNs).
   Guard: check `counts.sum(axis=1) == 0`; smoothing (add-α) is the fix,
   at the price of shrinking extreme probabilities toward uniform.
2. **Bucketing bias**: binning a continuous price into bands does NOT
   produce a Markov chain of regimes — persistence near bucket boundaries
   is manufactured by the discretization, not by the market (live demo:
   banded rows are valid stochastic matrices and still wrong as models).
   The honest statement: P̂ is only as Markov as your state definition
   (lesson 2.3 Axis 1 comes back to bite).

## 4. Key takeaways

- `N = (I−Q)⁻¹` is to absorbing chains what `P^k` is to ergodic ones: one
  matrix that answers every "how long / which end" question.
- MFPT by first-step analysis = Bellman without decisions or discounting —
  the linear-system skeleton that policy evaluation (2.2, 2.7) reuses.
- Estimating P from data is trivial to compute and easy to get silently
  wrong (zero rows, non-Markov buckets). Always verify the estimated chain
  reproduces held-out hitting statistics.

## Exercises

1. Gambler's ruin with p = 0.5: show E[time] = i·(N−i) — the closed form is
   quadratic; your N matrix should reproduce it to machine precision.
2. Credit migration: 3 states (AAA→default absorbing); compute expected
   time to default from AAA under your own rating matrix.
3. Add-1 smoothing to the bucketed price chain; measure how much of the
   persistence bias it removes vs the true generating model.
4. Kemeny's constant: `K = π·(trace of M)`... verify trace-related identity
   K = Σ_i π_i m_ii-with-return = Σ eigenvalue terms; interpret as the
   chain's single "random-destination commute" number.
