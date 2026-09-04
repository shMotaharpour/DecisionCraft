# DecisionCraft

Hands-on mastery of **MILP**, **Markov Decision Processes (MDP)**, and
**Reinforcement Learning (RL)** — with applications in finance, asset/risk
management, workforce & inventory planning, and game theory.

This repo backs a guided, compact course: each concept first appears as a
tiny from-scratch (miniature) implementation, then graduates to standard
industrial tooling on realistic scenarios.

## Roadmap

| Phase | Topic | Capstone Project |
|-------|-------|------------------|
| 1 | Linear & Integer Optimization (MILP) | Portfolio allocation under budget + farmland/warehouse layout planning |
| 2 | Markov Chains & MDP (stochastic dynamics) | Inventory ordering under uncertain demand |
| 3 | Reinforcement Learning (Q-learning → PPO) | Trading agent with risk control / simulator agent |
| 4 | Hybrid architectures & Game Theory | Integrated pipeline: RL/MDP strategic layer + MILP execution layer |

## Repository Layout

```
DecisionCraft/
├── phase1_milp/          # MILP: modeling alphabet, logic → linear constraints, HiGHS/OR-Tools
├── phase2_mdp/           # Markov chains, Bellman equations, Value/Policy Iteration
├── phase3_rl/            # Q-Learning, tabular methods, Gymnasium environments, PPO
├── phase4_hybrid/        # MILP + RL combos, game theory, capstone pipeline
├── notes/                # Theory digests (compact, applied — no proofs)
└── data/                 # Shared datasets & scenarios
```

## Conventions

- **Language:** all code, notes, and docs are in English.
- **Style:** miniature from-scratch examples first, then standard libraries.
- **Math:** applied depth only — intuition and equations, no formal proofs.
- **Tooling:** Python 3.11, NumPy, SciPy, HiGHS (highspy), OR-Tools, Pyomo,
  Gymnasium. Dependency management with `uv`.

## Getting Started

```bash
uv sync            # or: uv pip install -r requirements.txt
uv run pytest      # quick sanity checks
```
