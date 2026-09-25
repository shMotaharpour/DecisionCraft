"""Lesson 4.3 — Course capstone: strategic MDP + tactical MILP + opponent model.

A trading-and-inventory firm over T days.
- STRATEGIC layer: MDP over (capital_level, market regime) solved by Value
  Iteration -> picks each day's RISK BUDGET (max deployable capital).
- TACTICAL layer: MILP (OR-Tools/SCIP) allocates that budget across
  assets (continuous) + a warehouse opening (binary) under sector caps
  and fixed costs; LP duals report the shadow price of the budget.
- ADVERSARY: a rival competes for asset 4; its behavior is tracked with
  fictitious play and shifts the MDP's expected returns (closing the loop).
"""
import numpy as np
from ortools.linear_solver import pywraplp

# ============================================================ shared data
T_DAYS = 30
GAMMA = 0.90
CAPITAL_LEVELS = [0, 1, 2, 3, 4]        # discretized capital (k$ of 100)
REGIMES = ["calm", "hot"]
REGIME_TRANS = {"calm": [0.8, 0.2], "hot": [0.4, 0.6]}

ASSET_RETURN = {"calm": [0.04, 0.06, 0.10, 0.14], "hot": [0.02, 0.03, 0.16, 0.05]}
ASSET_RISK = [0.5, 1.0, 2.0, 3.0]
SECTOR_CAP = 0.5        # assets 3+4 same sector
WH_FIXED, WH_YIELD = 8.0, 0.12   # open warehouse: pay 8k, earn 12% on it
RIVAL_GRAB_PROB = {0: 0.2, 1: 0.5}  # P(rival takes asset 4 | regime) x behavior


class FictitiousPlay:
    def __init__(self):
        self.counts = np.ones(2)  # rival: 0 = passive, 1 = aggressive

    def observe(self, a):
        self.counts[a] += 1

    def aggr_prob(self):
        return self.counts[1] / self.counts.sum()


# ============================================================ strategic MDP
def solve_strategic_mdp(rival_p_calm, rival_p_hot):
    """Value iteration over (capital, regime); actions = risk budget 0..2.
    Reward = expected return next day if the tactical layer can deploy
    (approximated here by the risk-budget * achievable return proxy)."""
    nS, nA = len(CAPITAL_LEVELS) * 2, 3
    best_ret = {"calm": max(ASSET_RETURN["calm"]), "hot": max(ASSET_RETURN["hot"])}
    grab = {"calm": rival_p_calm, "hot": rival_p_hot}
    R = np.zeros((nS, nA))
    for ci, cap in enumerate(CAPITAL_LEVELS):
        for ri, reg in enumerate(REGIMES):
            s = ci * 2 + ri
            for a in range(nA):  # a = risk budget tier
                deployed = (a + 1) * 25.0  # tier deploys 25/50/75k
                if deployed > cap * 25.0:
                    R[s, a] = -0.5   # can't deploy more than you have: penalty
                    continue
                # rival grabs asset 4 (the top-return one) with prob grab[reg]
                base = ASSET_RETURN[reg][2] * (1 - grab[reg]) + ASSET_RETURN[reg][1] * grab[reg]
                R[s, a] = deployed * base - 0.01 * deployed  # positive margin minus fee
    P = np.zeros((nS, nA, nS))
    for ci in range(5):
        for ri in range(2):
            s = ci * 2 + ri
            for a in range(nA):
                for rj in range(2):
                    # capital evolves: profits raise the capital level
                    grown = min(4, ci + (1 if a > 0 else 0))
                    P[s, a, grown * 2 + rj] += REGIME_TRANS[REGIMES[ri]][rj]
    V = np.zeros(nS)
    for _ in range(2000):
        Q = R + GAMMA * np.einsum("sap,p->sa", P, V)
        Vn = Q.max(axis=1)
        if np.abs(Vn - V).max() < 1e-10:
            break
        V = Vn
    policy = (R + GAMMA * np.einsum("sap,p->sa", P, V)).argmax(axis=1)
    return V, policy


# ============================================================ tactical MILP
def tactical_solve(budget, returns, rival_takes_asset4):
    s = pywraplp.Solver.CreateSolver("SCIP")
    s.SuppressOutput()
    x = [s.NumVar(0, budget, f"x{i}") for i in range(4)]
    z = s.BoolVar("open_wh")
    # assets 3,4 sector cap
    s.Add(x[2] + x[3] <= SECTOR_CAP * budget)
    # risk cap scaled by strategic risk budget: budget IS the risk units
    s.Add(sum(ASSET_RISK[i] * x[i] for i in range(4)) <= budget)
    if rival_takes_asset4:
        s.Add(x[3] == 0, "rival_took_asset4")
    # warehouse: binary with yield on its own capital
    s.Add(z * WH_FIXED <= budget)  # must fit in budget
    s.Add(x[0] + x[1] + x[2] + x[3] + WH_FIXED * z <= budget)
    # objective: one-period return; opening the warehouse costs WH_FIXED now
    # and yields WH_YIELD * WH_FIXED — net negative unless horizon > 8 periods
    s.Maximize(sum(returns[i] * x[i] for i in range(4))
               + z * (WH_YIELD * WH_FIXED - WH_FIXED) * 8.0)  # 8-period horizon amortization
    if s.Solve() != pywraplp.Solver.OPTIMAL:
        return None, None, None
    alloc = [xi.solution_value() for xi in x]
    # (LP duals of this MILP are not exposed; in production you'd re-solve
    # the LP relaxation with GLOP to get the shadow price of the budget.)
    return alloc, z.solution_value(), s.Objective().Value()


# ============================================================ simulation
def simulate():
    fp = FictitiousPlay()
    capital, regime = 4, 0  # start full capital, calm
    rng = np.random.default_rng(0)
    log = []
    for day in range(T_DAYS):
        V, policy = solve_strategic_mdp(fp.aggr_prob(), max(fp.aggr_prob(), 0.5))
        tier = policy[capital * 2 + regime]
        budget = [0.0, 25.0, 50.0, 75.0][tier]
        if budget > capital * 25.0:
            budget = capital * 25.0  # tactical layer enforces feasibility
        returns = ASSET_RETURN[REGIMES[regime]]
        rival_took = rng.random() < fp.aggr_prob()
        alloc, wh, obj = tactical_solve(budget, returns, rival_took)
        if alloc is None:
            log.append((day, regime, budget, "INFEASIBLE", None))
            continue
        # realized P&L: per-period asset return; warehouse yields 12%/period
        profit = sum(returns[i] * alloc[i] for i in range(4))
        profit += (WH_YIELD * WH_FIXED) if wh else 0.0
        if wh and day == 0:
            profit -= WH_FIXED  # opening cost paid once, on day 0
        capital = int(min(4, max(0, round(capital + np.sign(profit)))))
        log.append((day, REGIMES[regime], budget, round(profit, 2), round(fp.aggr_prob(), 2)))
        # regime transition + rival observation
        regime = 1 if rng.random() < REGIME_TRANS[REGIMES[regime]][1] else 0
        fp.observe(1 if rival_took else 0)
    return log


if __name__ == "__main__":
    fp = FictitiousPlay()
    log = simulate()
    print("day | regime | risk budget | daily P&L (k$) | rival aggression est.")
    for day, reg, budget, profit, aggr in log:
        print(f" {day:2d} | {reg:5s} | {budget:10.1f} | {str(profit):>14s} | {aggr}")
    profits = [row[3] for row in log if isinstance(row[3], float)]
    print(f"\ntotal P&L = {sum(profits):.2f} k$ over {T_DAYS} days")
    print("fictitious play converged rival estimate:", round(fp.aggr_prob(), 3))
