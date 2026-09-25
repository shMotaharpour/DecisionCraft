# Phase 4 — Lesson 4.2: Repeated & Stochastic Games — Learning in Games

## 1. Why one-shot analysis is not enough

Lesson 4.1 solved a single simultaneous move. Real competition is
**repeated** (every trading day) and **sequential/stochastic** (the market
state evolves — exactly a Markov game, lesson 2.3). Repeated interaction
unlocks things impossible one-shot:

- **Punishment & reputation**: defect today, get punished tomorrow →
  cooperation becomes rational. **Folk theorems (proven)**: with enough
  patience (high discount factor γ), *any* individually-rational payoff
  profile can be an equilibrium of the repeated game. Consequence: the
  equilibrium *set* explodes — predicting behavior needs more than NE.
- **Trigger strategies** (grim trigger, tit-for-tat): simple, and in the
  repeated prisoner's dilemma tit-for-tat is famously robust.

## 2. Learning in games — opponents as unknown environments

Instead of assuming rationality, *learn* the opponent empirically — the
RL worldview applied to games:

| Method | Idea | Status |
|--------|------|--------|
| **Fictitious play** | play best response to the *empirical frequency* of opponent's past actions | converges to NE in zero-sum & potential games (proven); cheap, online |
| **Best-response dynamics** | alternately BR to last opponent action | can cycle (rock-paper-scissors!) — cycling is informative, not failure |
| **Exp3 / no-regret** | multiplicative-weights over actions; regret bound √(T·ln K) **proven** | adversarial opponents, no stationarity assumption |
| **Self-play RL** | train against (a copy of) yourself | AlphaGo/AlphaZero; in zero-sum with the right ingredients converges toward minimax — in general-sum, no such guarantee (proven gaps exist) |

**No-regret is the honest guarantee**: against ANY opponent, an algorithm
with regret o(T) does nearly as well as the best fixed action in
hindsight. In repeated zero-sum games, two no-regret players converge to
minimax equilibria (**proven** — the "regret matching" path to CFR).

## 3. Miniature experiment — repeated market-entry, three opponents

The script stages the lesson-4.1 game played 5,000 rounds between our
fictitious-play agent and three opponents:
1. a fixed script (always High),
2. an adaptive ε-greedy Q-learner (lesson 3.1!),
3. a copy of the agent itself (self-play).

Measured: final empirical strategy mix, average payoff, and — for
self-play — whether the play drifts toward the NE identified in 4.1.
Then the same with Exp3 weights to show the no-regret guarantee in action
against the *non-stationary* Q-learner opponent (where fictitious play's
stationarity assumption breaks).

## 4. Connection to the capstone

Kaggriculture-style competition = stochastic game + imperfect information.
The phase-4 architecture (lesson 4.3): predict opponent → best-respond
with a planner (MILP/DP). Fictitious play is exactly "estimate the
opponent model, then best-respond" — the theory behind the engineering.

## 5. Run it

```bash
uv run python phase4_hybrid/lesson4_2_repeated_games.py
```

## 6. Key takeaways

- Repetition changes equilibrium analysis: folk theorems make almost
  anything sustainable; punishment structure is the real content.
- Against adaptive opponents, **regret bounds replace equilibrium
  assumptions** — the only guarantee that survives non-stationarity.
- Fictitious play = RL (estimate) + optimization (best respond) — the
  hybrid pattern of lesson 4.3 in miniature.
- **Proven results cited:** folk theorems; fictitious-play convergence
  (zero-sum/potential); Exp3 √T regret; no-regret pairs converge in
  zero-sum (via minimax + regret decomposition).
