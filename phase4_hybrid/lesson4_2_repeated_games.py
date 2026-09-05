"""Lesson 4.2 — Repeated games: fictitious play, Exp3, self-play.

The lesson-4.1 market-entry game played 5,000 rounds vs three opponents:
fixed script, adaptive Q-learner, self (fictitious play mirror).
Then Exp3 (no-regret) vs the non-stationary Q-learner.
"""
import numpy as np

G_ROW = np.array([[6.0, 3.0],   # row payoff: G[row, col]
                  [8.0, 1.0]])
G_COL = np.array([[6.0, 8.0],   # col payoff
                  [3.0, 1.0]])
ROUNDS = 5_000
MAX_PAYOFF = 8.0


class FictitiousPlay:
    """Best-respond to the empirical action frequencies of the opponent."""

    def __init__(self):
        self.counts = np.ones(2)  # Laplace smoothing

    def act(self):
        opp_freq = self.counts / self.counts.sum()
        return int(np.argmax(G_ROW @ opp_freq))

    def observe(self, opp_action):
        self.counts[opp_action] += 1


class FixedHigh:
    def act(self):
        return 1

    def observe(self, a):
        pass


class QLearner:
    """Adaptive opponent from phase 3 (off-policy Q-learning on actions)."""

    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros(2)
        self.N = np.zeros(2)

    def act(self):
        if self.rng.random() < 0.10:
            return int(self.rng.integers(2))
        return int(np.argmax(self.Q))

    def observe(self, my_payoff, my_action):
        self.N[my_action] += 1
        self.Q[my_action] += 0.05 * (my_payoff - self.Q[my_action])


class Exp3:
    """No-regret multiplicative weights for adversarial bandits."""

    def __init__(self, gamma=0.07, seed=3):
        self.w = np.ones(2)
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self._last_p = np.array([0.5, 0.5])

    def act(self):
        p = (1 - self.gamma) * self.w / self.w.sum() + self.gamma / 2
        self._last_p = p
        return int(self.rng.choice(2, p=p))

    def observe(self, chosen, payoff):
        # importance-weighted loss estimate (payoff → loss in [0,1])
        p_chosen = self._last_p[chosen]
        loss = (MAX_PAYOFF - payoff) / MAX_PAYOFF
        x_hat = np.zeros(2)
        x_hat[chosen] = loss / p_chosen
        self.w *= np.exp(-self.gamma * x_hat / 2)


def play_round(a_row, a_col):
    return G_ROW[a_row, a_col], G_COL[a_row, a_col]


def duel(label, agent, opp):
    payoffs, my_actions = [], []
    for t in range(ROUNDS):
        a_row = agent.act()
        a_col = opp.act()
        r_row, r_col = play_round(a_row, a_col)

        # learning updates
        if isinstance(agent, FictitiousPlay):
            agent.observe(a_col)
        elif isinstance(agent, Exp3):
            agent.observe(a_row, r_row)
        elif isinstance(agent, QLearner):
            agent.observe(r_row, a_row)
        if isinstance(opp, QLearner):
            opp.observe(r_col, a_col)
        elif isinstance(opp, FictitiousPlay):
            opp.observe(a_row)

        payoffs.append(r_row)
        my_actions.append(a_row)

    payoffs = np.array(payoffs)
    actions = np.array(my_actions)
    print(f"{label:34s} last-1k avg payoff = {payoffs[-1000:].mean():6.3f}   "
          f"action mix (Low,High) = ({(actions[-1000:] == 0).mean():.2f}, "
          f"{(actions[-1000:] == 1).mean():.2f})")


print(f"repeated market-entry, {ROUNDS} rounds\n")
print("(a) fictitious play vs three opponents:")
duel("FP vs fixed-High", FictitiousPlay(), FixedHigh())
duel("FP vs adaptive Q-learner", FictitiousPlay(), QLearner())
duel("FP vs FP (self-play)", FictitiousPlay(), FictitiousPlay())

print("\n(b) Exp3 (no-regret) vs the non-stationary Q-learner:")
duel("Exp3 vs adaptive Q-learner", Exp3(), QLearner())

print("\nReference: zero-sum variant value = 3.0 at (Low, High) — lesson 4.1.")
print("FP vs fixed-High should learn to play Low (payoff 3 vs 1).")
print("Against a Low-heavy opponent the best response is High (8 vs 6).")
