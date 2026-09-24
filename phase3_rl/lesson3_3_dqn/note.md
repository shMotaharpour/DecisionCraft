# Phase 3 — Lesson 3.3: Function Approximation & DQN

## 1. Why the table must die

Tabular Q has one parameter per `(s, a)` pair. Count the cells:

| Problem | \|S\|×\|A\| |
|---------|------------|
| lesson 3.1 inventory | 11×11 = 121 |
| inventory with 3 correlated products | (11³)×11³ ≈ 1.8M |
| Kaggriculture one day | tiles×crops×actions ≈ 10⁷⁺ |
| FX order placement (book state ≈ 10⁴ levels, actions continuous) | ∞ |

Two killers, precisely:
1. **Memory**: you cannot store 10⁷⁺ floats.
2. **Generalization**: learning Q(s₁,a) teaches you *nothing* about
   Q(s₂,a) even when s₁≈s₂. Experience never transfers.

**Function approximation** replaces the table with `Q_θ(s, a) ≈ Q*(s, a)`
where θ is a parameter vector far smaller than the table. Now similar
states share parameters — learning transfers.

## 2. The deadly interaction: bootstrapping + non-linear FA + off-policy

With a linear `Q_θ = φ(s,a)ᵀθ` and *on-policy* data, TD converges
(**proven**, Tsitsiklis & Van Roy 1997). Break any of those three and the
plain TD update can diverge — the famous **deadly triad** (Sutton):
- bootstrapping (TD targets), 
- function approximation (non-linear),
- off-policy training.

DQN's entire architecture is a set of engineering patches around the triad.

## 3. DQN — the 2013/2015 recipe, component by component

| Component | Patch for | Mechanism |
|-----------|-----------|-----------|
| **Replay buffer** | correlated samples | store (s,a,r,s') tuples; train on random minibatches → breaks temporal correlation, reuses data (sample efficiency ↑) |
| **Target network** | moving targets | a frozen copy Q_θ⁻ computes TD targets; sync every N steps → target is stationary for N steps, kills the feedback loop |
| **Huber loss** | exploding gradients | quadratic near 0, linear beyond — robust to outlier TD errors |
| **Gradient clipping** | stability | cap the norm of the update |
| **ε-greedy decay** | exploration | lesson 3.2 |
| Reward/observation scaling | optimization | NNs hate raw magnitudes; normalize inputs, clip rewards |

**The loss:**
```
L(θ) = E_{(s,a,r,s')~buffer} [ Huber( r + γ·max_{a'} Q_{θ⁻}(s',a') − Q_θ(s,a) ) ]
```

**Proven-ish status (be honest here):** DQN has *no* general convergence
theorem. There are partial results (e.g. convergence of fitted-Q-iteration
under conditions; Neural Fitted Q analyses), and the 2015 Nature paper's
stability is empirical, not theoretical. DQN works because the patches
tame the triad *in practice*. Anything promising "guaranteed deep RL
convergence" should raise your suspicion.

## 4. Miniature DQN on the inventory problem

Same environment as 3.1/3.2 (state = inventory 0..10, actions 0..10) —
deliberately still tabular-sized so we can grade DQN against the exact
answer. State is one-hot encoded (teaches: even a "table" can be
re-expressed as a network; the machinery is what we're learning).

Success criteria: evaluated performance ≥ ~95% of exact; sanity-checks
(ε decay curve, loss decreasing, Q magnitudes sensible).

## 5. Run it

```bash
uv run python phase3_rl/lesson3_3_dqn_inventory.py
```

## 6. Key takeaways

- FA buys generalization at the price of the deadly triad; DQN = replay +
  target net + clipping, each patch mapped to a specific failure mode.
- **Linear FA + on-policy is provably safe** — when you can afford it
  (small feature sets, LSTD/LSPI), it's the honest choice; deep nets are
  for when you can't.
- Double DQN (decompose max into selection/evaluation), dueling, and
  prioritized replay are incremental patches you'll meet in real codebases;
  the triad remains the organizing principle.
- Next lesson (3.4): when actions are continuous or combinatorial,
  `max_a Q` itself becomes infeasible → policy gradients compute the
  gradient *with respect to the policy directly*.
