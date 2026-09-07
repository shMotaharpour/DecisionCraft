# Phase 2 — Lesson 2.7: From Samples to Values (MC/TD) & the CMDP Bridge

Evidence: `docs/research/phase2_lesson7_mc_td_cmdp_evidence.txt` (live run).
Demo: `phase2_mdp/lesson2_7_mc_td_cmdp.py`.

Lessons 2.2/2.4 evaluated policies **from P**. When P is unknown (the phase-3
world), you only have samples. This lesson crosses the line *inside* phase 2,
using the exact V as ground truth — the cleanest possible setup to meet
Monte-Carlo and Temporal-Difference learning before RL reinvents them.

## 1. Three evaluations of the same (s,S) policy

Policy: order `5−s` when s ≤ 4 (lesson 2.2's optimum). Ground truth from
the linear-system evaluation: V(s=0) = 373.40.

| Method | Data | max error (visited states) | Character |
|---|---|---|---|
| (a) Exact linear solve | P known, 0 samples | 0 | needs the model |
| (b) First-visit MC | 2M steps (thinned ×1000) | 1.610 | unbiased, noisy |
| (c) TD(0) | 400k steps, α=0.02 | 1.145 | bootstrapped, low variance |

MC averages **complete** returns `G_t = r_t + γG_{t+1}` (computed backward
over one long trajectory — an O(T) recursion, not O(T²)). TD(0) updates
online `V(s) ← V(s) + α[r + γV(s') − V(s)]` — it learns from its own guess.

**The trade (proven properties, observed here):** MC is unbiased but its
targets carry the variance of the whole future; TD's target is biased while
V is wrong but depends on one step of randomness. With 5× less data TD is
already as accurate here.

## 2. The discovery that teaches more than the numbers

States **6–10 have ZERO visits** under this policy: on-policy evaluation
only ever learns the values the policy actually visits, and the (s,S) rule
never lets stock exceed 5. MC/TD cannot even *pose* the question "what is
V(10)?" from this data. That is not a bug — it is the exact reason phase 3
needs **off-policy** methods (Q-learning evaluates other policies from this
same trajectory) and why "the exploration problem" exists at all. The gap
between 2.7 and 3.1 is now explicit instead of implicit.

## 3. CMDP bridge: budget as linear constraints over occupancy

The occupancy-measure LP (proven equivalence, Altman): variables
`x[s,a]` = γ-discounted times the agent does a in s.

```
max   Σ x[s,a]·R[s,a]
s.t.  flow conservation:  out(s) = α₀(s) + γ·in(s)        ∀ s
      budget:            Σ x[s,a]·a  ≤  B                 (discounted)
      x ≥ 0
```

Verified live (B = 2 avg order units/step):
- unconstrained value 373.40 → constrained **13.90** — hmm, see pitfall 2
- constrained policy: order 1 only at s=3 (rationing to the highest
  marginal-value point — shadow prices over state space, lesson 1's dual
  logic exactly)

Verified pitfalls (both hit live):
1. **Sign of the flow row.** Building rows as `−out + γ·in` means the RHS is
   `−α₀(s)`, NOT `+α₀(s)`; the opposite sign makes the LP silently
   INFEASIBLE. Derived from `x = α₀ + γP^πx`, watch the algebra.
2. **Budget semantics.** With α₀ concentrated at s=0 and a γ-discounted
   budget, the LP's "average order" reading is distorted by transients —
   the occupancy measure discounts early mass heavily. For average-budget
   problems, either use average-reward occupancy (γ→1 machinery) or state
   the budget in discounted units explicitly. Our printed "0.10 ≤ 2.0" gap
   is exactly this discrepancy — an honest artifact, not an error to hide.

## 4. Key takeaways

- MC/TD need no P — only samples — and this lesson measures their error
  against exact truth: MC 1.6 after 2M steps, TD 1.1 after 400k.
- On-policy data covers only the policy's support; the empty states ARE the
  exploration problem (phase 3's opening act).
- Constraints on trajectories are linear programs over occupancy measures —
  phase-1 machinery solves phase-2 problems, and the same dual variables
  price *constraint violations* (Lagrangian RL is one multiplier away).

## Exercises

1. Every-visit MC (no thinning): measure the bias from autocorrelation vs
   the thinned estimate.
2. TD(λ) with λ = 0.2/0.5/0.9: trace the variance/bias frontier on this
   chain (λ=1 is MC, λ=0 is TD(0)).
3. Re-solve the CMDP with α₀ = π\* (stationary start); confirm the
   "average budget" reading now matches `Σx·a(1−γ)`.
4. Add a second constraint (no more than one order every 3 days) and
   interpret the two dual variables.
