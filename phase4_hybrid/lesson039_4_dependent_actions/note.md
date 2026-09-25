# Phase 4 — Lesson 4.4: Action Spaces That Depend on State, History & Other Actions

The missing lesson — you spotted a real gap. "Can I buy this item?" depends
on the agent's level, wallet, and previous purchases. Almost no textbook
makes this explicit; every real environment has it.

## 1. First: whose problem is it? (MDP's or RL's?)

Answer: **it is fundamentally an MDP-definition problem; RL merely
inherits it (and adds engineering patches).**

- The MDP formalism already *contains* the answer: the transition kernel
  `P(s'|s,a)` is only defined for valid `(s,a)` pairs, and the formal
  object is a **state-dependent action set** `A(s) ⊆ A`.
  An action that is illegal in state s is simply *not in A(s)* — the
  transition "buy item while broke" does not exist. Bellman's argmax runs
  over `a ∈ A(s)`, and Value Iteration never even notices the issue.
  So conceptually: **formalize it in the MDP** (`A(s)`), not in the
  learning method.
- RL inherits it because practical RL algorithms want a *fixed-size*
  action head (softmax over k actions, Gaussian over ℝⁿ) — they fight
  the `A(s)` structure. Hence the engineering patches below.
- Game theory complicates it once more (lesson 4.2): if action
  availability depends on *history* (e.g. an item already sold to someone
  else), plain `A(s)` is insufficient — you must either fold history into
  the state (Markov enrichment, lesson 2.3) or accept a POMDP.

Three nested levels of "dependent actions":

| Level | Example | Formal fix |
|-------|---------|-----------|
| 1. State-dependent feasibility | "buy" needs wallet ≥ price | `A(s)` — mask infeasible actions |
| 2. Action-dependent actions | "upgrade" allowed only if "buy" happened first | encode prerequisites in s (flags) → back to level 1 |
| 3. History/global-resource dependence | limited shared stock, one-time offers | add resource/history to s; or POMDP if unobservable |

**The golden rule:** never let feasibility live only in the environment's
error handling. If an illegal action returns "error, −1 reward" and the
state doesn't change, you have polluted the MDP: the agent wastes samples
learning the trivially useless fact "illegal actions are bad," and worse,
off-policy methods (Q-learning) will still *bootstrap* those phantom Q
values into the max, distorting the policy. Mask instead.

## 2. The four engineering patterns (per algorithm family)

### (a) Action masking (value-based: DQN family)
Compute the set of feasible actions in state s; give impossible actions
Q = −∞ before the argmax. In torch: add a large negative bias (≈ −1e9) to
the logits of infeasible actions *before* softmax/argmax. Critically, the
TD target's `max_a'` must use the **same mask in state s'** — masking only
in selection but not in the target is the classic bug (your agent learns
to value phantom actions it can never take).

### (b) Action domains (CP-SAT)
`NewIntVarFromDomain` naturally expresses "actions available right now";
rebuild-per-step (lesson 1.4d) means each step's model simply has a
different domain. Feasibility logic can *itself* be constraints:
`price <= wallet` as a model constraint rather than a pre-filter.

### (c) Big-M / indicator constraints (MILP)
`buy_i ≤ level_i · y` — purchase enabled by level binary; this is lesson
1.2's toolbox verbatim. For "at most k purchases total": `Σ y_i ≤ k`.

### (d) Policy-side handling (policy gradients: PPO family)
Mask in the softmax (invalid logits → −inf) so the policy assigns them zero
probability; then `log_prob` of sampled (always feasible) actions is fine.
Entropy and the ratio stay well-defined. Alternative: parameterize a
variable-length action head, or factorize "which action type" × "which
parameters" (autoregressive policies — how game AI agents emit
(type, target, amount) triples).

### Credit assignment with prerequisites (the subtle one)
If "upgrade" requires a prior "buy", the *return* of the buy action must
include the future value unlocked — mask/credit tricks don't do this
automatically. Options: reward shaping (small immediate credit for
enabling moves — beware shaping bias, *proven*: potential-based shaping
Ng et al. 1999 preserves optimal policies), or rely on the discounting to
propagate it (slow but unbiased).

## 3. Run the demos

```bash
uv run python phase4_hybrid/lesson4_4_dependent_actions.py
```

Three miniature demos, one per family:
1. **DQN with masking** on a tiny level/shop env — masked vs unmasked
   (error-penalty) training, showing the unmasked agent wasting samples
   and distorting Q.
2. **CP-SAT** with `NewIntVarFromDomain` + prerequisite constraints —
   per-step action domains.
3. **PPO-style masked softmax** — invalid actions get zero probability;
   feasibility mask computed from state.

## 4. Key takeaways

- `A(s)` is an MDP concept — the formalism always supported it; RL adds
  masking engineering.
- **Mask both the selection AND the bootstrap target** (Q-learning) —
  the single most common dependent-action bug.
- Prefer masking over "error penalties"; prefer state-enrichment over
  history-masking when feasible; use potential-based shaping if you must
  shape credits for prerequisite chains (proven safe).
- In Kaggriculture terms: "plant" requires tile EMPTY + water + seed stock
  + level — that is a per-step mask over tiles, computed from state, not a
  fixed action list.
- **Proven results cited:** potential-based reward shaping preserves
  optimality (Ng, Harada, Russell 1999); masking preserves the MDP's
  optimal policy (it only removes actions outside A(s)).
