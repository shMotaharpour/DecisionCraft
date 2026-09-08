# Phase 3 — Lesson 3.7: Capstone — RL vs Rule-Based vs Exact DP on a Tiny Farm

Evidence: `docs/research/phase3_lesson7_farm_capstone_evidence.txt` (live run).
Demo: `phase3_rl/lesson3_7_farm_capstone.py`.

The phase-3 finale: a MINIATURE of Kaggriculture (the game the Chista agent
plays) — one tile, 12 steps, 3 crops with different (cost, price, growth
time), water mechanics, stochastic rain. Everything phase 3 built is
deployed; and because the game is small, we can do what real RL projects
usually cannot: **compute the exact optimum and grade everyone against it.**

## 1. The env (`FarmEnv`)

- State: crop planted (4-way), watered flag, budget (0..6), weather, step
  one-hot → 19-dim observation vector.
- Actions: noop / plant crop-1..3 / water / harvest.
- Weather: 30% rain (free watering). Growth requires water every step.
- Sparse economics: only harvest and invalid actions pay; water is a small
  negative; selling pays 5/8/14 for crops costing 1/2/3 (growth 2/3/5).

## 2. The exact ceiling — backward induction, again (lesson 2.6!)

State space: `13 t × 4 crop × 6 growth × 2 watered × 31 budget = 19,344`
states (budget on the 0.2 lattice — all costs are multiples of 0.2).
`lru_cache`-memoized recursion solves it in **0.2 s**: **V\* = 22.25**.

This is the whole course in one table row: phase 2's machine (2.6 finite
horizon) prices the game exactly; phase 3's agents chase that number from
samples. RL papers call this "upper bound from an oracle" — here the oracle
costs 0.2 s.

## 3. The three-way comparison (3000-episode eval, live)

| policy | avg reward | vs exact |
|---|---|---|
| **exact DP** | 22.248 | 100% |
| **PPO (3000 eps)** | **15.157** | **68%** |
| rule-based | 13.558 | 61% |
| DQN (2000 eps) | 9.589 | 43% |

**RL beat the hand-written policy**: PPO's 68% > rule-based 61% — the
learned policy found budget-management the greedy rule lacks (the rule
wastes the episode tail where crop-3's short cycle would pay; PPO discovered
cycle-planning from data alone). This is the capstone's thesis: on YOUR
game, RL can beat the hand rule — and the exact DP tells you precisely how
much room is left (32%).

DQN's 43% is the sparse-reward lecture: replay+target learn the watering
loop but under-explore the harvest timing in 2000 episodes. Reward sparsity
hits value methods harder than policy methods here — a concrete, live
instance of the DQN-vs-PPO trade.

## 4. Verified live lessons

1. **Entropy coefficient matters** (0.01 → policy collapses to one action;
   0.02 → water/harvest appear). 3.2's exploration lesson in PG clothing.
2. **Sparse rewards hit DQN harder than PPO** (43% vs 68%): only harvest
   pays, ~1-in-12 steps. Bootstrapped value propagates slowly when the
   paying transition is rare and late.
3. The gap to the DP optimum decomposes into *exploration* (finding the
   harvest moment) × *credit assignment* (which of the 11 earlier actions
   earned it). This capstone is the testbed where you SEE both, because the
   exact answer is known.
4. **Rule-based is a floor, not a ceiling**: PPO found cycle-planning the
   hand rule lacks — but note the 3000-episode budget was needed; at 800
   episodes PPO was still at 7.5 (below the rule). Sample efficiency is
   the real cost of "learning instead of writing".

## 5. Key takeaways

- Whenever your "RL problem" can be made small enough for exact DP — do it
  FIRST. The oracle gives every later experiment a target and a diagnosis.
- The rule-based baseline is the minimum bar: RL that can't beat a
  hand-written policy on its own game is not ready (the Chista lesson:
  the MILP+prediction hybrid exists because pure RL was below it).
- Sparse rewards + long horizons = the frontier where phase-3 methods show
  their age. Everything in phase 4 (hybrid architectures) is a response.
- **The Chista connection**: the rule-based vs PPO vs DP ladder is exactly
  the project's architecture ladder (hand rules → learned policy → exact
  optimization), and why the production agent is hybrid.

## Exercises

1. Reward-shape the water step (small +0.05 for watering a growing crop)
   and re-train: how much of the DQN gap closes? (And what does it do to
   optimality vs the DP answer?)
2. Extract the DP's policy table and DIFF it against rule_policy — list
   every state where the hand rule is wrong. That list is your next
   feature/rule iteration.
3. Double the horizon to 24 steps: DP is still instant; re-run PPO 3×
   longer. Report both curves — the computational asymmetry IS the hybrid
   argument for phase 4.
4. Make rain probability state-dependent (higher after sun) and show the
   Markov property breaks for the old observation vector; fix the state.
