"""Lesson 4.5 — POMDP deep dive: censored-demand inventory.

The classic hidden-state trap: you observe SALES = min(D, stock); demand
beyond a stock-out is invisible (censored).

(a) Bayesian belief over the Poisson rate lambda, updated with the
    CENSORED likelihood — including the key insight that a zero-sale day
    at zero stock is EVIDENCE OF HIGH demand.
(b) head-to-head: naive MLE agent vs belief agent (both plan with VI on
    their own estimated model), same demand stream.
(c) value-of-information: as starting stock grows, censoring (and the
    belief agent's advantage) shrinks.
"""
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize

TRUE_LAMBDA = 4.0        # hidden: true mean demand
PRICE, COST, HOLD = 10.0, 3.0, 0.5
MAX_ORDER = 12
GAMMA = 0.95
DAYS = 200
STOCK_CAP = 15


def true_demand(rng):
    return rng.poisson(TRUE_LAMBDA)


# ---------------------------------------------------- (a) Bayes filter on λ
GRID = np.linspace(0.5, 10.0, 96)   # discretized belief over λ


def bayes_update(belief, sales, stock):
    """P(sales=k | λ, stock): censored Poisson likelihood.
    k < stock  -> P(D = k)
    k == stock -> P(D >= stock)   (stock-out: demand >= stock, hidden!)"""
    lik = np.where(
        GRID > 0,
        [poisson.pmf(sales, lam) if sales < stock else poisson.sf(stock - 1, lam)
         for lam in GRID],
        1e-12,
    )
    post = belief * lik
    return post / post.sum()


print("(a) Bayesian belief tracking under censoring")
rng = np.random.default_rng(0)
belief = np.ones(len(GRID)) / len(GRID)
stock = 2  # deliberately tight -> lots of censoring
trace = []
for day in range(60):
    d = true_demand(rng)
    sales = min(d, stock)
    belief = bayes_update(belief, sales, stock)
    stock = min(STOCK_CAP, max(0, stock - sales + 4))  # naive refill to 4
    if day in (0, 9, 29, 59):
        post_mean = (belief * GRID).sum()
        trace.append((day, sales, post_mean))
print("  day | sales | posterior mean of λ")
for day, sales, mean in trace:
    print(f"  {day:3d} | {sales:5d} | {mean:.3f}")
print(f"  (true λ = {TRUE_LAMBDA}; posterior converges despite censoring)")


# ---------------------------------------------------- (b) two agents
def plan_with_lambda(lam):
    """VI on the inventory MDP with a *known* lambda (phase-2 machinery)."""
    S = np.arange(STOCK_CAP + 1)
    pmf = poisson.pmf(np.arange(STOCK_CAP + 1), lam)
    pmf[-1] += 1 - pmf.sum()
    P = np.zeros((len(S), MAX_ORDER + 1, len(S)))
    R = np.full((len(S), MAX_ORDER + 1), -1e9)
    for s in S:
        for a in range(MAX_ORDER + 1):
            if s + a > STOCK_CAP:
                continue
            R[s, a] = sum(pr * (PRICE * min(s + a, d) - HOLD * max(s + a - d, 0)
                                - COST * a) for d, pr in enumerate(pmf))
            for d, pr in enumerate(pmf):
                P[s, a, min(STOCK_CAP, max(0, s + a - d))] += pr
    V = np.zeros(len(S))
    while True:
        Q = R + GAMMA * np.einsum("sap,p->sa", P, V)
        Vn = Q.max(axis=1)
        if np.abs(Vn - V).max() < 1e-9:
            break
        V = Vn
    return np.where(R > -1e8, R, -np.inf).argmax(axis=1), V


class NaiveAgent:
    """Estimates lambda from observed sales only (ignores censoring)."""

    def __init__(self):
        self.sales_sum, self.days = 0.0, 0

    def act(self, stock):
        lam = max(0.3, self.sales_sum / max(self.days, 1))
        self.policy, _ = plan_with_lambda(lam)
        return int(self.policy[stock])

    def observe(self, sales, stock):
        self.sales_sum += sales
        self.days += 1


class BeliefAgent:
    """Maintains posterior over lambda with the censored likelihood; plans
    on the posterior mean (thesis: belief-summary policy)."""

    def __init__(self):
        self.belief = np.ones(len(GRID)) / len(GRID)

    def act(self, stock):
        lam = (self.belief * GRID).sum()
        self.policy, _ = plan_with_lambda(lam)
        return int(self.policy[stock])

    def observe(self, sales, stock):
        self.belief = bayes_update(self.belief, sales, stock)


def run_agent(agent, seed, start_stock=2, refill=0):
    """Daily loop; agent orders each day; reward = standard inventory P&L.
    refill: fixed extra supply that reduces censoring (info experiment)."""
    env_rng = np.random.default_rng(seed)
    stock = start_stock
    total, stockouts = 0.0, 0
    for day in range(DAYS):
        order = agent.act(stock)
        order = min(order, STOCK_CAP - stock)
        d = true_demand(env_rng)
        sales = min(stock + order, d)
        r = (PRICE * min(stock + order, d) - COST * order
             - HOLD * max(stock + order - d, 0))
        total += r
        stock = min(STOCK_CAP, max(0, stock + order - d))
        if stock == 0:
            stockouts += 1
        agent.observe(sales, stock)
        stock = min(STOCK_CAP, stock + refill)
    return total, stockouts


print("\n(b) head-to-head on identical demand streams (seed-matched)")
naive = NaiveAgent()
belief = BeliefAgent()
r_n, so_n = run_agent(naive, seed=42)
r_b, so_b = run_agent(belief, seed=42)
print(f"  naive MLE agent : total P&L = {r_n:8.1f}, stockouts = {so_n}")
print(f"  belief agent    : total P&L = {r_b:8.1f}, stockouts = {so_b}")
print(f"  belief advantage: {r_b - r_n:+.1f} ({100*(r_b-r_n)/abs(r_n):+.1f}%)")

# ---------------------------------------------------- (c) value of information
print("\n(c) value of information: advantage shrinks as censoring shrinks")
print("  refill | naive P&L | belief P&L | advantage")
for refill in (0, 2, 4, 8):
    rn, _ = run_agent(NaiveAgent(), 42, refill=refill)
    rb, _ = run_agent(BeliefAgent(), 42, refill=refill)
    print(f"  {refill:6d} | {rn:9.1f} | {rb:10.1f} | {rb - rn:+.1f}")
print("  (more exogenous supply -> fewer stock-outs -> less censored data")
print("   -> the hidden-state information advantage decays, as theory predicts)")
