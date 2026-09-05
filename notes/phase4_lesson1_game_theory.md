# Phase 4 — Lesson 4.1: Game Theory Foundations — From Optimization to Interaction

## 1. The conceptual jump

Phases 1–3 optimized against *nature* (probabilities). Game theory
optimizes against an **adaptive opponent who is optimizing against you**.
One line changes everything: "what is best for me" stops being a number
and becomes a *function of the other player's choice*.

## 2. Normal-form games — the complete vocabulary

A game = players, strategy sets, payoffs `u_i(strategy_i, strategy_-i)`.

| 2-player payoff structure | Name | Example |
|---------------------------|------|---------|
| sum = 0 always | **zero-sum** (strictly competitive) | market share split, Kaggriculture ranking |
| sum ≠ 0 | **general-sum** | negotiations, oligopoly pricing |

Key equilibrium concepts (all *proven* to exist):
- **Dominant strategy**: best regardless of opponent. Rare in practice.
- **Nash equilibrium (NE)**: a strategy profile where no player gains by
  unilateral deviation. **Nash's theorem (1950): every finite game has at
  least one NE (in mixed strategies)** — proven via Kakutani/Brouwer
  fixed points. Existence ≠ uniqueness ≠ computability-in-polynomial-time
  (best-NE is PPAD-hard — a proven complexity result that matters!).
- **Minimax = maximin**: in **zero-sum** games, `max min = min max` —
  **von Neumann's minimax theorem (1928)**, proven via LP duality!
  The value of the game is unique. This is THE bridge to phase 1.

## 3. The LP duality miracle (why phase 1 owns game theory)

Zero-sum game with payoff matrix M (row player maximizes):

```
Row:  max v   s.t.  Σ_i M[i,j]·p_i ≥ v  ∀j,  Σp_i = 1, p ≥ 0
Col:  min w   s.t.  Σ_j M[i,j]·q_j ≤ w  ∀i,  Σq_j = 1, q ≥ 0
```

These two LPs are **duals**. By strong duality (**proven**): optimal
values coincide (the game value), and the optimal strategies p*, q* are
exactly the minimax mixed strategies. Solving a zero-sum game = solving
one LP. Mixed strategies are not behavioral noise — they are the
*rational optimum*, and the simplex method proves it.

## 4. Miniature game — market entry (computed live)

Two firms choose capacities {Low, High} simultaneously; profits
counter-cyclical (one's High crowds the other's margin). The script
computes: best-response correspondences, pure NEs, the mixed equilibrium
of the zero-sum variant via LP (scipy linprog / OR-Tools), and verifies
the minimax value equals the maximin value.

## 5. Beyond one-shot (preview of lesson 4.2)

- **Repeated games**: equilibria multiply (folk theorems — *proven*);
  reputation and punishment become rational. Pricing wars, trading
  relationships.
- **Extensive form / imperfect information**: trees + information sets
  (poker = Kuhn's theorem, CFR algorithm).
- **Stochastic games** (Shapley 1953 — met in lesson 2.3): Markov games;
  RL opponents. Kaggriculture is one.
- **Mechanism design**: reverse game theory — design the rules
  (auctions: revenue equivalence, *proven*).

## 6. Run it

```bash
uv run python phase4_hybrid/lesson4_1_games_lp.py
```

## 7. Key takeaways

- Zero-sum ⇒ minimax theorem ⇒ one LP solves the game (**proven chain**).
- General-sum: NE exists but finding/best-selecting is PPAD-hard —
  respect the complexity, use iterative methods (4.2).
- Mixed strategies are optimal behavior under adversarial unpredictability
  — directly relevant to execution algos that must be unpredictable to
  front-runners.
- Von Neumann → Shapley → Nash is the same mathematical thread that phase
  2's minimax MDP sits on.
