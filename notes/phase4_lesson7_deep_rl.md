# Phase 4 — Lesson 4.7: Deep Learning Meets RL/MDP/POMDP — The Full Story

Why neural networks changed RL, why they also broke it, and the complete
toolkit — architectures, instabilities, and tricks — that makes Deep RL
work in practice. This is the bridge lesson between "tabular with
guarantees" (phases 2–3) and "industrial deep RL" (the real world).

## 1. What deep learning actually buys (and what it costs)

### The buy
- **Generalization over state space** (lesson 3.3): a network shares
  parameters between similar states — 10⁵⁰ states, 10⁶ parameters.
- **End-to-end perception**: pixels/board images/order books → decisions,
  no hand-crafted features (DQN playing Atari from raw pixels, 2015).
- **Memory for POMDPs**: recurrent heads build internal belief-like state
  from observation history (lesson 4.5, family (c)).
- **Non-linear value/policy classes**: expressive enough for real
  dynamics (markets, physics, games).

### The cost — why "just use a bigger network" fails
Tabular RL had **proven convergence** (Watkins, Tsitsiklis–Van Roy).
Deep RL traded the guarantees away. The four structural diseases:

1. **The Deadly Triad** (lesson 3.3, now in its full horror):
   bootstrapping × non-linear function approximation × off-policy data
   → divergence. Q-values can explode to 10⁶⁰ (observed in practice).
2. **Non-stationarity from the agent's own updates**: the data
   distribution depends on the policy, the policy depends on the network,
   the network just changed → the learning problem shifts under its feet
   (contrast supervised learning's fixed i.i.d. dataset).
3. **Plasticity loss**: networks trained on shifting targets gradually
   lose the ability to keep learning (weights saturate); TabulaRasa/
   continual-learning literature documents it as a first-class problem.
4. **Sparse, delayed, noisy rewards**: credit assignment across thousands
   of steps of self-generated experience — nothing like a labeled dataset.

**Honest status: Deep RL has no general convergence theorem.** What
exists is a large, battle-tested engineering canon, organized below.

## 2. The architecture catalog (what to use, when)

### 2.1 Value-based family
| Architecture | Idea | When |
|--------------|------|------|
| **DQN** (2015) | CNN + replay + target net | discrete actions, pixels |
| **Double DQN** (2016) | decouple action *selection* (online net) from *evaluation* (target net) | fixes DQN's chronic overestimation bias (**proven analysis** of the bias) |
| **Dueling DQN** | split head into V(s) + Advantage(s,a): A(s,a)=Q−V | states where action choice barely matters |
| **C51 / QR-DQN** | distributional heads (lesson 4.6) | richer signal, risk read-outs |
| **Rainbow** (2018) | all of the above + prioritized replay + n-step + noisy nets | the reference value-based agent |
| **R2D2** | + LSTM, stored recurrent state in replay, burn-in | partial observability + value-based |

### 2.2 Policy-based family
| Architecture | Idea | When |
|--------------|------|------|
| **A2C/A3C** | actor-critic, parallel envs | simple, decent baseline |
| **PPO** (2017) | clipped trust region, GAE | the default everywhere, incl. RLHF |
| **SAC** (2018) | off-policy + **maximum-entropy** objective: max E[r] + αH(π) | continuous control (robotics); sample-efficient; entropy = principled exploration (lesson 3.2) |
| **TD3** | twin critics (min of two Q's) + delayed policy update | fixes SAC/DQN overestimation in continuous control |
| **DDPG** | deterministic policy gradient | older; TD3 is its repair |

### 2.3 Model-based family (learn the world, plan inside it)
| Architecture | Idea | When |
|--------------|------|------|
| **World Models** (2018) | VAE (perceive) + RNN (predict) + controller | imaginative pre-training |
| **Dreamer v1-v3** | latent imagination: learn dynamics in latent space, backprop through imagined rollouts | sample efficiency king in many benchmarks |
| **MuZero** (2019) | like AlphaZero but *learns* the dynamics model; MCTS in latent space | games + planning; no known rules needed |
| **EfficientZero** | + self-supervised value equivalence | sample efficiency near model-free SOTA/100 |

### 2.4 POMDP-specific architectures
| Architecture | Mechanism |
|--------------|-----------|
| **DRQN / frame-stack** | CNN over k last frames (poor man's memory) |
| **LSTM/GRU heads** (R2D2, IMPALA) | learned internal state — the practical belief approximator |
| **Transformer policies** (Gato, Decision Transformer) | sequence modeling over (obs, act, rew) history; RL-as-transformer |
| **World models again** | the latent state IS the belief; Dreamer de-facto solves POMDPs by reconstruction |

### 2.5 Offline RL (learning from logs — the finance case)
You rarely can explore live with real money. Offline RL = learn from a
fixed logged dataset, no environment interaction:
- **CQL** (conservative Q-learning): penalize OOD action Q values,
  **provable** lower bounds on true Q;
- **BCQ / IQL**: restrict to dataset-supported actions;
- **Decision Transformer**: sequence modeling, no Bellman at all.
This is the honest paradigm for trading backtests: exploration in
production = blowing up the fund.

## 3. The stabilization trick canon (why each exists)

| Trick | Disease it treats | Mechanism |
|-------|-------------------|-----------|
| **Experience replay** | correlated samples, data waste | uniform/minibatch sampling from a big buffer |
| **Target networks** (slow-moving copy) | moving bootstrap targets | freeze parameters τ steps (DQN) or Polyak-avg θ' ← τθ+(1−τ)θ' (SAC/TD3) |
| **Gradient clipping / Huber** | exploding TD errors | cap updates |
| **Reward/obs normalization & clipping** | optimization pathologies | standardize inputs; clip Atari rewards to [−1,1] |
| **Prioritized replay (PER)** | uniform sampling wastes rare informative transitions | sample ∝ \|TD error\| with importance weights (proven bias correction) |
| **n-step returns / GAE(λ)** | bias-variance dial of the TD target | λ-trace; GAE is THE variance knob of PPO |
| **Entropy bonus / temperature** | premature determinism | soft policies, exploration (SAC's α auto-tuning) |
| **Noisy nets** | ε-greedy's amnesia | parametric noise in layers → *state-conditional* exploration (learned curiosity) |
| **Learning-rate schedules, Adam, layer-norm** | plasticity loss | keep networks trainable for millions of steps |
| **Action masking** (lesson 4.4!) | invalid-action probability mass | −1e9 logits pre-softmax |
| **PopArt / value scaling** | value-scale drift across tasks | normalized value heads with denormalization |
| **Distributional heads** | weak scalar signal | per-atom targets (lesson 4.6) |

## 4. What the theory still gives you (be precise about this)

- **Approximation error bounds** (proven): with features φ in a bounded
  class, fitted value iteration is stable *if* you project (Bellman
  residual methods, ALP — lesson 2.4). Deep nets = features you don't
  control → bounds exist only in restricted regimes (e.g. linear-in-
  features analyses of deep nets, NTK theory).
- **Overestimation analyses** (proven): max-of-noisy-estimates is
  biased upward (the exact defect Double DQN / TD3 repair).
- **Coherent risk + distributional contraction** (proven, lesson 4.6).
- **No-regret and policy-gradient theorems** (lessons 3.2, 3.4).
- Everything else is empirics. The correct engineering stance:
  borrow the canon, verify on YOUR environment, keep the tabular/
  linear versions as sanity baselines (this course's habit).

## 5. Practical recipe (the one I'd actually deploy)

1. Start with the simplest thing that could work: tabular → linear FA
   (lesson 2.4) → PPO defaults. Measure; only escalate when stuck.
2. Discrete actions: Rainbow components, in order of measured value:
   prioritized replay > distributional > double > dueling > noisy.
3. Continuous actions: SAC (or TD3), not PPO, when sample efficiency
   matters; PPO when simplicity/sim-parallelism dominates.
4. Partial observability: add an LSTM/GRU + R2D2-style stored state;
   verify the memory actually helps by ablating observation history.
5. From logs only: CQL/IQL, never naive Q-learning offline (it
   hallucinates on unvisited actions — lesson 4.4's phantom-Q problem,
   amplified).
6. Always: action masking, reward scaling, normalized observations,
   several seeds (deep RL's seed variance is brutal — report mean±std
   over ≥5 seeds or don't report), and the exact-policy sanity check
   whenever a smaller solvable model of your problem exists.

## 6. Run the demos

```bash
uv run python phase4_hybrid/lesson4_7_deep_rl_demos.py
```

Demonstrations (kept small enough to run on CPU):
1. **Overestimation bias live**: tabular Q-learning vs Double Q-learning
   on a stochastic bandit-with-context — measured overestimation of the
   max operator (the defect Double DQN repairs).
2. **Replay buffer vs on-the-fly**: sample efficiency and stability of
   DQN-style training with/without replay on the inventory task.
3. **Target network necessity**: training divergence without the frozen
   copy (deadly triad caught in the act).
4. **LSTM memory for POMDP**: the censored-demand env (lesson 4.5), now
   solved by a GRU policy over observation history vs a memoryless MLP —
   showing learned belief beats forgetting.
