# Phase 3 — Lesson 3.1: RL Foundations — Learning When You Can't Plan

## 1. The phase boundary, precisely

Phase 2 needed `P(s'|s,a)` written down. Real problems deny us that:
- demand distributions unknown or non-stationary;
- state spaces too big to enumerate transitions for;
- the only access to reality = *interact and observe*.

RL = the family of methods that find good policies **from experience
alone**. The MDP formalism (lesson 2.2) survives unchanged — only the
*solution machinery* changes: sampling replaces the model.

## 2. The three families of learning signal

| Family | Update signal | Bias/Variance | When it wins |
|--------|--------------|---------------|--------------|
| **Monte Carlo (MC)** | actual full-episode return G_t | unbiased, high variance | episodic tasks, easy to wait for the end |
| **TD(0) / SARSA / Q-learning** | bootstrapped r + γV(next) | biased, low variance | continuing tasks, sample efficiency |
| **TD(λ) / n-step** | λ-mixed returns | tunable middle | the practical dial |

**Proven anchors:** TD(0) converges to V^π under standard step-size
conditions (Robbins–Monro); Q-learning converges w.p.1 to Q\* if every
(s,a) is visited infinitely often and Σα = ∞, Σα² < ∞ (Watkins 1989).
No model needed — that theorem *is* the license for phase 3.

## 3. The tabular trio, concretely

All three maintain a table `Q[s, a]`; they differ in the target:

```
MC:       Q[s,a] ← Q[s,a] + α ( G_t − Q[s,a] )
SARSA:    Q[s,a] ← Q[s,a] + α ( r + γ Q[s', a'] − Q[s,a] )     ← on-policy
Q-learn:  Q[s,a] ← Q[s,a] + α ( r + γ max_{a'} Q[s', a'] − Q[s,a] )  ← off-policy
```

- **On-policy (SARSA)** evaluates the policy you actually follow — safer
  with risky actions (it learns the value of *exploring*).
- **Off-policy (Q-learning)** learns the greedy optimum regardless of how
  you behave — the data can come from any behavior policy. This separation
  (behavior π_b ≠ target π) is what later enables replay buffers and
  learning from logs — the exact setup of real trading backtests.
- **Expected SARSA** uses Σ_a π(a|s')Q(s',a) — lower variance than
  Q-learning's max.

## 4. The same inventory problem — but model-blind

We reuse lesson 2.2's inventory MDP, but the agent now:
- does NOT know the Poisson rates, prices, or the transition kernel;
- only sees (s, a, r, s') tuples by acting.

Success criterion: tabular Q-learning's greedy policy ≈ the exact (s,S)
policy from phase 2 (V\* known). This is the *only* honest way to evaluate
RL code: compare against the exact answer on a problem small enough to
have one.

## 5. Run it

```bash
uv run python phase3_rl/lesson3_1_qlearning_inventory.py
```

The script: builds the environment (same economics, hidden model), trains
MC / SARSA / Q-learning with identical exploration, plots-style-compares
learning curves, and prints all three final policies vs the exact (s,S)
rule from lesson 2.2.

## 6. What to watch in the output (and why it matters)

1. **Q-learning and SARSA converge to nearly the same (s,S) policy** —
   in this environment exploration is not dangerous, so on/off-policy
   differences vanish. In risky environments (margin trading!) they don't.
2. **MC converges slowest** — high variance per update.
3. **Learning curves are non-monotonic** — exploration occasionally
   degrades the current best policy. Policy stability ≠ value stability.
4. Step size α decays like 1/√(visits) — satisfying the proven
   convergence condition while staying responsive early.

## 7. Bridge to the next lessons

- **Where does the sample→value bridge come from?** Lesson 2.7 already
  compared MC vs TD(0) against exact V on this very policy — this phase
  re-derives it with the model hidden.
- Exploration *how* is lesson 3.2 (ε-greedy is the weakest answer; the
  bandit theory behind it is lesson 3.5).
- When |S| explodes or states are continuous, the table dies —
  function approximation + deep nets is lesson 3.3 (DQN).
- When the *argmax* over actions is itself hard or continuous —
  policy gradients and PPO is lesson 3.4.
- The phase lands on its capstone, lesson 3.7: RL on a mini farm game,
  graded against exact DP.
