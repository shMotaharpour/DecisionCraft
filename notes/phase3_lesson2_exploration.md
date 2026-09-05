# Phase 3 — Lesson 3.2: Exploration — The Explore/Exploit Dilemma

## 1. The dilemma, stated sharply

An agent that always exploits freezes on the first good-looking action and
never discovers better ones. An agent that always explores never banks the
rewards it finds. This is not an engineering nuisance — it is a
**fundamental theorem-shaped tradeoff** (no algorithm can be simultaneously
sublinear-regret and always-greedy; exploration has an unavoidable cost,
*proven* in the multi-armed bandit regret lower bound, Lai–Robbins 1985:
any policy must suffer regret ≥ log(T)·Σ(μ\*−μᵢ)/KL(μᵢ,μ\*)).

Exploration matters far beyond RL: A/B testing, new-product vs proven
product, re-balancing vs holding — all are bandits.

## 2. The toolbox (from weakest to strongest)

### (a) ε-greedy
With probability ε pick a uniformly random action. Simple, robust — and
the weakest: it explores *every* action equally, even the obviously
terrible ones, forever.
- **Decay schedules** fix the "forever" part: linear, exponential, or
  1/√t. Decay too fast → premature convergence on a suboptimal policy
  (lesson 3.1's frozen policies); too slow → wasted samples forever.
- **ε-greedy with optimism**: initialize Q optimistically (e.g. max
  possible reward); then pure greedy *is* explorative until each action's
  estimate falls to realistic levels — exploration becomes systematic.

### (b) Optimism in the face of uncertainty (UCB)
Instead of a random coin, quantify *how uncertain* each action's value is
and pick the action with the best **upper confidence bound**:

```
a_t = argmax_a [ Q(a) + c · sqrt( ln t / N(a) ) ]
```

- `Q(a)` = exploit term; the sqrt term = exploration bonus, large when
  the action has been tried rarely (N(a) small).
- **Proven:** UCB1 achieves logarithmic regret — matches the theoretical
  optimum up to constants. The bonus is Hoeffding's inequality (proven)
  turned into a decision rule.
- Deep RL descendant: UCB appears inside tree search (MCTS, AlphaGo's
  selection step) as `Q + c_puct·P·sqrt(ΣN)/(1+N)`.

### (c) Posterior sampling / Thompson sampling
Keep a Bayesian belief over each action's value; sample an action by
*drawing from the posterior* and acting greedily on the sample. Elegant,
empirically excellent, and the natural bridge to distributional RL.

### (d) Entropy bonuses (policy methods, lesson 3.4)
Instead of value-side exploration, keep the *policy distribution* entropic:
add `β·H(π)` to the objective. This is how PPO/A2C explore.

### (e) Curiosity / count-based bonuses (state-space exploration)
When states (not actions) are the bottleneck: bonus for visiting rarely-seen
states. In the inventory world: "try ordering when the warehouse is nearly
full" must sometimes happen, or Q[state≈MAX, a>0] stays unvisited garbage
forever.

## 3. Why this lesson is not just ε-greedy

The inventory problem (lesson 3.1) has a nasty property for exploration:
**visits concentrate**. Under a good policy the system lives in s≈0..5,
so Q[8..10, ·] get few samples and stay noisy; a *one-off* demand shock
can make a never-visited action look brilliant. Sparse-visit + drifting
targets = the two conditions where naive ε-greedy decay fails. The
experiment below demonstrates exactly this.

## 4. Run it

```bash
uv run python phase3_rl/lesson3_2_exploration.py
```

The script runs the inventory problem with four exploration strategies —
constant ε, decayed ε, optimistic init, UCB1-adapted — and reports for
each: final evaluated performance, how well Q is estimated in RARELY
visited states (the Q-error at s=10), and total regret vs the exact policy.

## 5. Key takeaways

- Exploration cost is *provable* and unavoidable (regret lower bounds);
  the question is only which logarithmic-constant you pay.
- ε-greedy decays = "explore now, exploit later" — fine when the world is
  stationary and the action space is small.
- **Optimism/UCB = explore what you're uncertain about** — the principled
  version, and the ancestor of MCTS.
- In finance, exploration = deliberately taking small losing positions to
  learn about instruments; UCB's "bonus proportional to uncertainty" is
  precisely a rational research budget.
- **Proven results cited:** Lai–Robbins lower bound (1985); UCB1 log-regret
  (Auer et al. 2002); Hoeffding's inequality backing the bonus term.
