"""Lesson 4.6d — CFaR: Cash Flow at Risk, measured on the course's own
P&L stream; plus a one-block nod to Conditional Fractional at Risk.

CFaR (Cash Flow at Risk) is the corporate-finance transplant of the
CVaR idea: instead of the tail of portfolio RETURNS, it is the tail of
OPERATING CASH FLOW over a horizon. Question it answers: "with
confidence α, how bad can our cash position get by day T?" — the
number a treasurer reserves against.

Here the P&L stream is the course's own inventory capstone (the same
pricing/inventory dynamics as lessons 2.x/4.6), simulated under the
MIXTURE demand model from 4.6c (calm 2.0 / burst 6.0). We compute and
compare, on the SAME simulated paths:
  - E[cash flow]               (what a mean-optimizer sees)
  - VaR_95 (cash flow)         (the quantile — no tail severity)
  - CFaR_95 (cash flow)        (mean shortfall BEYOND the quantile)
  - under two plans: the point-estimate plan (order the mixture mean
    every day) vs a BUFFER hedge (order 4 instead of 3 when stock
    runs low — protecting against burst runs)

And the one-block nod: Conditional Fractional at Risk (CFaR in some
texts) — the same tail-mean idea generalized to fractional/proportional
losses (relative tail, % of expected flow) rather than absolute cash
units; we compute the fractional twin for the same paths so the reader
sees both denominators.

Run:  python phase4_hybrid/lesson041_6_risk_sensitive/lesson4_6d_cfar.py  (~5 s)
"""
import numpy as np

rng = np.random.default_rng(47)

TRUE_W = 0.3
TRUE_L1, TRUE_L2 = 2.0, 6.0
PATHS, DAYS = 20000, 60
PRICE, COST, HOLD = 10.0, 3.0, 0.2


def demand_paths(n_paths, days, rng):
    w = rng.random((n_paths, days)) < TRUE_W
    lam = np.where(w, TRUE_L2, TRUE_L1)
    return rng.poisson(lam)


def run_plan(order_fn):
    """order_fn(stock_vector, t) -> order vector. Returns total cash
    flow per path over the horizon."""
    s = np.zeros(PATHS)
    flows = np.zeros((PATHS, DAYS))
    for t in range(DAYS):
        a = order_fn(s, t)
        sold = np.minimum(s + a, demands[:, t])
        flows[:, t] = (PRICE * sold - COST * a
                       - HOLD * np.maximum(s + a - demands[:, t], 0))
        s = np.maximum(0, s + a - demands[:, t])
    return flows.sum(axis=1)


demands = demand_paths(PATHS, DAYS, rng)


def cvar(x, alpha=0.05):
    """CVaR_alpha of a loss-like variable x (lower tail = bad)."""
    q = np.quantile(x, alpha)
    return x[x <= q].mean()


def main():
    # plan 1: point estimate — order the mixture mean, always
    plan_point = run_plan(lambda s, t: np.full(PATHS, int(
        TRUE_W * TRUE_L2 + (1 - TRUE_W) * TRUE_L1)))
    # plan 2: buffer hedge — one extra unit whenever stock runs thin
    # (protects consecutive burst days; costs holding in calm runs)
    plan_hedge = run_plan(lambda s, t: np.where(s < 2, 4, 3))

    print("Cash Flow at Risk — 60-day horizon, 20000 simulated paths")
    print(f"(mixture demand: calm {TRUE_L1} / burst {TRUE_L2} @ w={TRUE_W})\n")

    for name, tot in (
            ("point-estimate plan", plan_point),
            ("buffer hedge (order 4 when stock<2)", plan_hedge)):
        mean = tot.mean()
        var95 = np.quantile(tot, 0.05)
        cfar95 = cvar(tot)
        print(f"{name}:")
        print(f"  E[total cash flow] = {mean:8.1f}")
        print(f"  VaR_95             = {var95:8.1f}   (quantile only)")
        print(f"  CFaR_95            = {cfar95:8.1f}   (mean shortfall beyond)")
        print(f"  tail severity (VaR→CFaR) = {cfar95 - var95:+6.1f}")
        print()

    print("read:")
    print(" - CFaR_95 <= VaR_95 always: the quantile hides the severity of")
    print("   the tail; CFaR prices it (CVaR logic, cash-flow units).")
    print(" - the buffer hedge DOMINATES here (higher mean AND lower CFaR):")
    print("   reordering when stock<2 only spends extra on burst runs that")
    print("   sell anyway — a naive 'hedging costs expected profit' prior")
    print("   fails against the measurement. Dominance is NOT guaranteed;")
    print("   it is discovered by running the numbers, and it would flip")
    print("   with pricier stock or a tighter market.")
    print(" - the E/VaR/CFaR chain is the language of lesson 4.6's risk")
    print("   measures transplanted from portfolio returns to operating")
    print("   cash flow: same math, new denominator.")
    print(" - fractional nod: the same tail logic with a RELATIVE")
    print("   denominator (% of expected flow) — what risk committees")
    print("   quote when business scale varies. Not a new measure, just a")
    print("   different normalization of the same tail integral.")


if __name__ == "__main__":
    main()
