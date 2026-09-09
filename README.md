# DecisionCraft

Hands-on mastery of **MILP**, **Markov Decision Processes (MDP)**, and
**Reinforcement Learning (RL)** — with applications in finance, asset/risk
management, workforce & inventory planning, and game theory.

This repo backs a guided, compact course (37 lessons): each concept first
appears as a tiny from-scratch (miniature) implementation, then graduates
to standard industrial tooling on realistic scenarios. Every lesson has a
runnable demo; claims that can be measured are measured (evidence files
under `docs/research/`).

## Course map

### Phase 1 — Linear & Integer Optimization (MILP)

| Lesson | Topic | Script |
|---|---|---|
| 1.1 | The modeling alphabet — toy portfolio | `lesson1_1_portfolio_ortools.py`, `lesson1_1_scipy.py` |
| 1.2 | Logic → linear constraints (fixed costs, cardinality) | `lesson1_2_logic_*.py` |
| 1.2b | Ordered selection + symmetry breaking + alternative optima | `lesson1_2b_order_symmetry.py` |
| 1.2c | General integer variables; parameter vs variable | `lesson1_2c_integer_vars.py` |
| 1.3 | Capstone: portfolio + warehouse MILP, shared budget | `lesson1_3_capstone.py` |
| 1.3b | Task assignment: Hungarian vs MILP, skills, min-max fairness | `lesson1_3b_task_assignment.py` |
| 1.4 | OR-Tools pywraplp API tour (7 categories) | `lesson1_4_ortools_api_tour.py` |
| 1.4b | CP-SAT complete constraint reference (8 categories) | `lesson1_4b_cpsat_tour.py` |
| 1.4c | Callbacks + constraint cost hierarchy | `lesson1_4c_callbacks_cost.py` |
| 1.4d | Model reuse, build cost, symmetry handling | `lesson1_4d_reuse_cost_symmetry.py` |
| 1.5 | Routing library: TSP → CVRP → VRPTW → P&D | `lesson1_5a_routing_tour.py` |
| 1.6 | Metaheuristics from scratch vs GLS; exact-vs-heuristic crossover | `lesson1_6a_metaheuristics.py`, `lesson1_6b_bignumber_milp.py` |
| 1.7 | LNS (destroy & repair); CP-SAT vs GLS scale test | `lesson1_7a_lns.py`, `lesson1_7b_scale_cpsat_vs_routing.py` |
| 1.8 | Scheduling (CP-SAT intervals), flows, bin packing | `lesson1_8a_scheduling.py`, `lesson1_8b_flows.py`, `lesson1_8c_binpacking.py` |
| 1.9 | Dantzig-Wolfe decomposition & column generation | `lesson1_9_dantzig_wolfe.py` |
| 1.10 | Branch-and-Price (CG inside branch-and-bound) | `lesson1_10_branch_and_price.py` |

### Phase 2 — Markov Chains & MDP

| Lesson | Topic | Script |
|---|---|---|
| 2.1 | Markov chains: simulation, P^k forecasts, stationary distribution | `lesson2_1_markov_chain.py` |
| 2.2 | MDP + Bellman: Value Iteration vs Policy Iteration, (s,S) discovery | `lesson2_2_mdp_inventory.py` |
| 2.3 | The complete MDP taxonomy (6 axes) + Kaggriculture classification | — |
| 2.4 | Beyond tabular: curse of dimensionality, ALP, MDP-as-LP, MPC | `lesson2_4_approx_alp.py` |
| 2.5 | Absorbing chains, hitting times, fitting P from data | `lesson2_5_absorbing_fit.py` |
| 2.6 | Finite-horizon backward induction (seasonal markdown) | `lesson2_6_finite_horizon.py` |
| 2.7 | MC/TD policy evaluation vs exact V; CMDP occupancy LP | `lesson2_7_mc_td_cmdp.py` |
| 2.8 | Design tradeoffs I: time & state resolution, measured | `lesson2_8a_time_granularity.py`, `lesson2_8b_state_granularity.py` |
| 2.9 | Design tradeoffs II: reward shaping & terminal value | `lesson2_9a_reward_design.py`, `lesson2_9b_terminal_value.py` |
| 2.10 | Design tradeoffs III: constraint pricing & MPC deployment | `lesson2_10a_constraint_penalty.py`, `lesson2_10b_precompute_mpc.py` |

### Phase 3 — Reinforcement Learning

| Lesson | Topic | Script |
|---|---|---|
| 3.1 | Tabular RL: MC / SARSA / Q-learning, model-blind | `lesson3_1_qlearning_inventory.py` |
| 3.2 | Exploration: ε schedules, optimism, UCB1 — measured failures | `lesson3_2_exploration.py` |
| 3.3 | DQN: replay buffer, target network, Huber loss | `lesson3_3_dqn_inventory.py` |
| 3.4 | REINFORCE → Actor-Critic → PPO on a Gymnasium env | `lesson3_4_ppo_gymnasium.py` |
| 3.5 | Bandits: regret theory, UCB1 vs Thompson, Lai–Robbins verified | `lesson3_5_bandits.py` |
| 3.6 | The deadly triad, experimentally (DQN ablations) | `lesson3_6_triad_ablation.py` |
| 3.7 | Capstone: RL vs rule-based vs exact DP on a mini farm | `lesson3_7_farm_capstone.py` |

### Phase 4 — Game Theory & Hybrid Architectures

| Lesson | Topic | Script |
|---|---|---|
| 4.1 | Normal-form games, minimax via LP duality | `lesson4_1_games_lp.py` |
| 4.2 | Repeated games: fictitious play, Exp3, self-play | `lesson4_2_repeated_games.py` |
| 4.3 | Capstone: strategic MDP + tactical MILP + opponent model | `lesson4_3_capstone_hybrid.py` |
| 4.4 | Dependent/parallel actions: masking across all families | `lesson4_4_dependent_actions.py` |
| 4.5 | POMDPs: belief states, censored-demand trap | `lesson4_5_pomdp.py` |
| 4.6 | Risk-sensitive & distributional RL (CVaR policy selection) | `lesson4_6_risk_sensitive.py`, `lesson4_6a_qrdqn.py`, `lesson4_6b_bayes.py` |
| 4.7 | Deep RL architecture catalog + stabilization canon | `lesson4_7_deep_rl_demos.py` |
| 4.8 | Markov games: Shapley VI, Markov fictitious play | `lesson4_8_markov_games.py` |
| 4.9 | Monte Carlo Tree Search from scratch (the AlphaGo engine) | `lesson4_9_mcts.py` |

Figures: `assets/phase{1..4}/` — 22 seeded PNGs, each regenerated by a
`make_figures*.py` script next to the lesson that owns it and embedded
in that lesson's note (after the numbers it illustrates). Mermaid
diagrams (4.3 architecture, 4.7 decision flow) live inline in the notes.
Evidence: `docs/research/` — one file per measured batch (19 files).

## Repository layout

```
DecisionCraft/
├── phase1_milp/          # MILP: modeling alphabet, OR-Tools (pywraplp/CP-SAT/Routing/flows), scheduling, metaheuristics, LNS, DW/CG, B&P
├── phase2_mdp/           # Markov chains, Bellman/VI/PI, taxonomy, ALP, absorbing chains, finite horizon, MC/TD, CMDP
├── phase3_rl/            # tabular RL, exploration, DQN, PPO, bandits, triad ablation, farm capstone
├── phase4_hybrid/        # game theory, repeated games, hybrid capstone, POMDP, risk, deep RL, Markov games, MCTS
├── notes/                # theory digests (compact, applied — no proofs), one per lesson
├── assets/               # figures (PNG) per phase, embedded in the notes
├── docs/research/        # evidence files: raw live outputs backing every measured claim
└── data/                 # shared datasets & scenarios
```

## Conventions

- **Language:** all code, notes, and docs are in English.
- **Style:** miniature from-scratch examples first, then standard libraries.
- **Math:** applied depth only — intuition and equations, no formal proofs;
  theorems used are labeled *proven* so the reader knows what may be relied on.
- **Tooling:** Python 3.11, NumPy, SciPy (`scipy.optimize.milp` / `linprog`),
  OR-Tools 9.15 (pywraplp, CP-SAT, Routing, graph flows), Gymnasium,
  PyTorch (CPU wheel). Dependency management with `uv`.
  (Pyomo/highspy were removed: unused — scipy's MILP uses HiGHS as its
  engine; `docs/research/` holds the raw evidence outputs.)
- **Honesty rules:** every measured claim cites an evidence file; negative
  results are reported, not tuned away; figures are seeded and reproducible.

## Getting started

```bash
uv sync
uv run python phase1_milp/lesson1_1_portfolio_ortools.py   # lesson 1.1 demo
```

Lesson notes live in `notes/phase<N>_lesson<M>_<slug>.md`; start at 1.1 and
follow the bridges each note ends with.
