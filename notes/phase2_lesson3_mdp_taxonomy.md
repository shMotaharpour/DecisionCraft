# Phase 2 — Lesson 2.3: The Complete MDP Taxonomy

A comprehensive map of MDP variants. Each axis below changes *what tools
apply* — this is the checklist you run through BEFORE writing any solver
code, on any decision problem (trading, inventory, farming, games).

## Axis 1 — What kind of state? (S)

| State type | Name | Solvable by | Example |
|------------|------|-------------|---------|
| Finite set, small | **Finite/tabular MDP** | VI/PI exact (lesson 2.2) | inventory level 0..10 |
| Finite set, huge | **Large tabular** | tabular methods die; approximate VI, or RL | board games (chess ~10⁴⁷) |
| Continuous vector | **Continuous-state MDP** | discretize, or kernel/LFA methods | portfolio weights, cash amount |
| Mixed discrete+continuous | **Hybrid MDP** | mostly RL + function approximation | (regime ∈ {bull,bear}, cash ∈ ℝ) |
| Partially observed | **POMDP** | belief-state MDP over filtered states | order book true state, hidden opponent hand |
| Structured / factored | **Factored MDP** | DBN decompositions, LP on factors | farm with independent tiles |

**Design rule (from lesson 2.1, now precise):** the state must be
**Markov** w.r.t. the real process. If reality needs memory, the fix is
always *state enrichment*: append the missing summary (e.g. last k
returns, moving averages, current position) until the Markov property
holds approximately. Under-treating this causes silent failure: your
"optimal" policy is optimal for the wrong process.

## Axis 2 — What kind of action? (A)

| Action space | Name | Notes |
|--------------|------|-------|
| Finite small | discrete actions | argmax over A in Bellman — easy |
| Finite huge / combinatorial | **combinatorial actions** | argmax itself becomes a MILP! (RL ⇄ OR bridge, lesson 4) |
| Continuous 1-D | continuous actions | argmax needs optimization; policy-gradient shines |
| Continuous high-D | continuous vector actions | DDPG/PPO-style RL; Bellman argmax impractical |
| Multi-agent (others adapt) | **game** | Markov game / stochastic game — opponent's policy is part of the environment; Nash replaces "optimal" |

Key insight: **the argmax inside Bellman is itself an optimization
problem.** If `A` is "which subset of 50 tiles to plant", the inner argmax
is a knapsack — and the MDP solver becomes a solver-of-solvers. This is
the main bridge between phase 2 and phase 4 (hybrid MILP+DP architectures).

## Axis 3 — Time horizon & discount

| Formulation | Value definition | Used when |
|-------------|------------------|-----------|
| **Finite horizon T** | `V_t(s) = max_a R + E V_{t+1}` — backward induction over t, time-dependent V | known end (trading day, harvest season, match) |
| **Infinite horizon, discounted** | `V(s) = max_a R + γ E V(s')`, γ<1 | stationary economics, default in RL |
| **Infinite horizon, average reward** | maximize long-run per-step average | steady-state operations; no γ tuning |
| **Episodic with terminal states** | absorbing state with 0 future reward | games that end, liquidation |

Practical equivalences (all *proven* results): finite-horizon with large T
≈ discounted with γ = (1−1/T); average-reward = limit of discounted as
γ→1 with the right scaling (Blackwell optimality). The **episodic trick**:
add an artificial absorbing "game over" state to convert any episode into
an infinite-horizon problem — that's exactly how Kaggriculture's 24-step
days map onto this math.

## Axis 4 — What do we know about the dynamics?

| Knowledge level | Paradigm |
|-----------------|----------|
| P(s'|s,a) known in closed form | **planning** — VI/PI exact (lesson 2.2) |
| P unknown, simulator available | **model-free RL** (phase 3) — sample transitions |
| P unknown, must also be estimated | **model-based RL** — learn P̂ then plan inside it |
| P known but too big to store | **approximate planning** — rollout, Monte Carlo Tree Search |

## Axis 5 — Reward structure

| Structure | Consequence |
|-----------|-------------|
| Bounded, γ<1 | convergence guaranteed |
| Unbounded (e.g. wealth) | use relative/utility transform or risk math breaking |
| Risk-sensitive | replace E with E−λ·Var, or exponential utility — changes Bellman into non-linear but still tractable forms |
| Constraints ("max drawdown ≤ X") | **Constrained MDP** → Lagrangian reward shaping or LP-on-occupancy-measures (connects to phase 1!) |
| Multi-objective | scalarize or Pareto-scan |

## Axis 6 — Determinism of the transition

- **Deterministic** (P is 0/1): Bellman collapses to shortest-path on a
  graph — Dijkstra/A* territory; MDP machinery is overkill.
- **Stochastic**: full MDP.
- **Adversarial** (environment picks worst case, not sampled): the
  **minimax MDP** / robust MDP — value = max_a min_{s'} …; this is the
  formal bridge to game theory (phase 4). Between "sampled by nature" and
  "chosen by adversary" lies a continuum; choosing where your problem sits
  is a *modeling decision* with big consequences (optimistic vs robust).

## The decision checklist (use before coding)

1. Is the state Markov? What must I append to it?
2. Continuous or discrete? How big is S×A in actual cells?
3. Finite or infinite horizon? Which discount?
4. Is P known / simulable / neither?
5. Is the environment indifferent, or adversarial (→ game)?
6. Are there constraints on the trajectory (→ CMDP)?
7. Is the inner argmax trivial or a hard combinatorial problem (→ hybrid)?

## Where Kaggriculture sits on every axis (worked classification)

| Axis | Kaggriculture answer |
|------|---------------------|
| State | finite but combinatorial (tiles × crops × resources) → factored |
| Actions | combinatorial per step (plant/water/harvest several tiles) |
| Horizon | episodic, fixed-length match (days × 24 steps) + terminal reward |
| Dynamics | partially known: rules known, opponent + some randomness not |
| Adversarial? | yes, 2-player → stochastic game, not plain MDP |
| Constraints | budget, water, tool cooldowns → CMDP flavor |
| Inner argmax | hard: which tiles to act on → the MILP bridge |

That's why the Chista project's MILP+prediction architecture was the
right engineering call — and why phase 4 will formalize it.

The escape routes this taxonomy promises are built in later lessons:
absorbing chains & hitting times (2.5), finite-horizon backward
induction (2.6), and the sample→value bridge with a CMDP worked LP
(2.7) — each covers one axis's "how" for the cases marked above.

**Proven results cited in this lesson:** Bellman optimality for all four
horizon formulations; Blackwell optimality (γ→1); contraction/convergence
only where γ<1 (average-reward needs different proofs); minimax values of
zero-sum stochastic games exist and are computable by value iteration
(Shapley 1953).
