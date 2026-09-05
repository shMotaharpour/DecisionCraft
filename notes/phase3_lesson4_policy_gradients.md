# Phase 3 — Lesson 3.4: Policy Gradients — REINFORCE → Actor-Critic → PPO

## 1. Why value-based methods hit a wall

DQN needs `max_a Q(s,a)`. That breaks in two real cases:
1. **Continuous actions** (how many shares to buy: a = 1,247.5?):
   argmax over ℝⁿ has no closed form; enumerating is impossible.
2. **Combinatorial actions** (which subset of 40 tiles to plant):
   the inner argmax is itself a hard MILP — DQN degenerates into
   solve-a-MILP-per-step.

**Policy-based methods skip the argmax entirely**: parameterize the policy
`π_θ(a|s)` directly and follow the gradient of expected return.

## 2. The policy gradient theorem (the proven foundation)

```
∇_θ J(θ) = E_{π_θ} [ ∇_θ log π_θ(a|s) · Q^π(s,a) ]
```

**Proven** (Policy Gradient Theorem, Sutton et al. 2000; Williams 1992 for
REINFORCE). Read it: "increase the log-probability of actions in
proportion to how good they turned out." No argmax, no importance weights,
works for any differentiable π_θ — categorical, Gaussian, whatever.

## 3. The three rungs of the ladder

### REINFORCE (1992) — the pure Monte-Carlo gradient
```
for each episode:  G_t = discounted return from t
                   θ ← θ + α · G_t · ∇ log π_θ(a_t|s_t)
```
Unbiased (uses true returns) but **high variance** — whole-episode noise
rides on every update. Needs many episodes.

### Variance reduction: the three standard tricks
1. **Baseline**: subtract b(s) from G_t; any action-independent b keeps
   the gradient unbiased (**proven**) but cuts variance massively.
   b(s)=V(s) → the term becomes the *advantage* A = Q − V.
2. **Causality**: only future rewards credit an action (G_t, not G_0).
3. **Entropy bonus**: add β·H(π_θ) to prevent premature determinism
   (exploration from lesson 3.2, policy-side).

### Actor-Critic
Learn V(s) with a second network (the **critic**) via TD; use the
bootstrapped advantage A(s,a) = r + γV(s') − V(s) as the REINFORCE weight.
Lower variance than MC, biased (bootstrap) — the classic tradeoff.
**A2C/A3C** = this, parallelized.

### PPO (2017) — the industry workhorse
Actor-critic + a *trust region*: don't let one update move the policy too
far from the data-collecting policy. Clipped surrogate:

```
ratio ρ = π_θ(a|s) / π_θ_old(a|s)
L = E[ min( ρ·A,  clip(ρ, 1−ε, 1+ε)·A ) ]
```

Why clipping matters (**proven intuition from TRPO's theory**): a large
ratio × large advantage = a destructive update; the clip bounds the
per-sample influence. PPO = simple (no second-order constraints like TRPO),
stable, parallelizes, and is the default in RLHF for LLMs — the same
algorithm that tuned ChatGPT. Multiple epochs over the same rollout batch
(sample efficiency) with the clip protecting against over-optimization.

## 4. Gymnasium — the interface that makes RL portable

```python
obs, info = env.reset()                    # start episode
obs, reward, terminated, truncated, info = env.step(action)  # act
```
`terminated` = true terminal state; `truncated` = time limit (the
episodic-trick distinction from lesson 2.3!). Any env implementing this
protocol plugs into PPO implementations unchanged — we build a
Gymnasium-native inventory env in the script and train PPO on it with a
from-scratch minimal implementation (no stable-baselines3 magic), so every
moving part stays visible.

## 5. Run it

```bash
uv run python phase3_rl/lesson3_4_ppo_gymnasium.py
```

Compares: REINFORCE vs Actor-Critic vs PPO on the Gymnasium inventory env,
all graded against the exact policy. Watch REINFORCE's noisy curve vs
PPO's stability.

## 6. Key takeaways

- Value-based (DQN) breaks on continuous/combinatorial actions;
  policy gradients remove the argmax — **the proven Policy Gradient
  Theorem is the license**.
- REINFORCE → baseline/advantage → actor-critic → PPO = one continuous
  story of variance reduction and trust regions.
- PPO's clip = bounded policy change per update; the same idea powers
  RLHF. You now know its mechanics from first principles.
- Gymnasium's terminated-vs-truncated split is not pedantry: bootstrapping
  on a truncation is a classic silent bug.
- Phase-3 capstone next (lesson 3.5): RL agent on YOUR game, simplified.
