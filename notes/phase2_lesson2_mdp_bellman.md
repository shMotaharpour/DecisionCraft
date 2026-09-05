# Phase 2 — Lesson 2.2: MDP — Adding Actions to the Chain

## 1. From chain to decision problem

A Markov chain evolves passively. An **MDP** gives *you* a lever: at each
time step you pick an **action** `a`, and the transition probabilities
depend on it. The 5-tuple:

```
(S, A, P, R, γ)
S: states        A: actions
P(s'|s,a): transition probability under action a
R(s,a): immediate reward        γ ∈ [0,1): discount factor
```

`γ` prices the future: a reward in k steps is worth `γ^k`. Two readings:
economically it's a discount rate, mathematically it guarantees the value
sum converges (**proven**: with bounded rewards and γ<1, the infinite sum
Σ γᵗ·r_t converges absolutely).

## 2. The miniature problem — single-product inventory (newsvendor-ish)

State: `s` = inventory at start of the day, `s ∈ {0..MAX}`.
Action: `a` = order amount, `0 ≤ a ≤ MAX − s`.
Overnight demand `D ~ Poisson(λ)`. Economics:

```
reward(s, a, D) = p·min(s+a, D)   − h·max(s+a−D, 0)   − c·a
                  sold revenue      holding cost       order cost
```
p=10, h=0.5, c=3, λ=3, MAX=10, γ=0.95.

Crucially, the *chain* part now has an action-conditional transition
matrix: demand randomness still makes `P(s'|s,a)` stochastic, but YOU
choose which row family to be in.

## 3. The Bellman equation — the heart of everything

For the optimal value function V(s) = best achievable discounted reward:

```
V(s) = max_a [ R(s,a) + γ · Σ_{s'} P(s'|s,a) · V(s') ]
```

Read it: "my value today = best immediate reward + discounted value of
where randomness takes me tomorrow." The `max` is the whole difficulty —
it makes the equation non-linear in V.

Two *proven* facts you can lean on:
- **Contraction mapping:** the Bellman optimality operator is a γ-contraction
  → it has a unique fixed point V\*, and Value Iteration converges to it
  geometrically (error ≤ γᵏ·span).
- **Policy improvement:** given any policy's V, acting greedily w.r.t. V
  is never worse; Policy Iteration terminates in finitely many steps
  (finitely many policies, strictly improving).

## 4. The two exact algorithms

**Value Iteration (VI):** iterate `V ← T(V)` (apply the Bellman operator)
until the change is < ε·(1−γ)/γ. Dead simple, anytime.

**Policy Iteration (PI):** (1) policy evaluation: solve the *linear* system
`V = R_π + γ P_π V` (no max → linear algebra!); (2) policy improvement:
act greedily w.r.t. V; repeat until the policy stops changing. Usually
fewer outer iterations, each heavier.

## 5. Run it

```bash
uv run python phase2_mdp/lesson2_2_mdp_inventory.py
```

The script: builds P(s'|s,a) and R(s,a) from the Poisson demand, runs VI
and PI, shows they agree on both V\* and the optimal policy π\*, and
prints the resulting ordering rule.

## 6. Key takeaways

- MDP = chain + decisions. P is now per-action; the optimal policy is a
  mapping `s → a`, and its value function is the fixed point of Bellman.
- VI = fixed-point iteration on a contraction; PI = Newton-like method on
  the policy space. Both are **exact** — no learning, no samples, full
  model knowledge required.
- The inventory optimum comes out as a *policy*: "when below X, order up
  to Y" — an (s, S) rule. The solver discovered the business rule.
- **Limitation that motivates RL (phase 3):** VI/PI need `P(s'|s,a)` in
  closed form. When demand distributions are unknown, or the state is
  combinatorial (a whole farm/portfolio), we can't write P down — we need
  to *learn from interaction*. That's RL.
- Bellman's curse of dimensionality: state space grows exponentially with
  dimensions. All of approximate RL lives in that shadow.
