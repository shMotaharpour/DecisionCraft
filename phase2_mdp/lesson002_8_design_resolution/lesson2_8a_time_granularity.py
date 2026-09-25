"""Lesson 2.8a — Time granularity: one problem, three clocks.

The SAME markdown/pricing problem (seasonal item, fad-decaying
price-sensitive Poisson demand) solved at three decision-epoch lengths:
  weekly (6 epochs)  — coarse: fewer decisions, less control
  2-day  (12 epochs) — middle
  daily  (42 epochs) — fine: most control, biggest state space
Same calendar, same demand process, same price menu. Each epoch, the
planner picks a price for the WHOLE epoch; demand accrues daily inside
it. Measures: optimal expected profit, |S|, solve time. The tradeoff is
real: finer clocks buy repricing options, and the price is |S| growth.

Run: python phase2_mdp/lesson2_8a_time_granularity.py   (~20 s)
"""
import time

import numpy as np
from scipy.stats import poisson

# --- shared problem data (one 6-week season = 42 days) -----------------
DAYS = 42
LAM_BASE = 4.0              # demand rate at reference price, day 0
PRICE_REF = 10.0
ELAST = 0.35                # demand multiplier = (p_ref/p)^ELAST
COST = 3.0                  # unit cost (paid at sale, cogs accounting)
HOLD_PER_DAY = 0.08         # per unit per DAY
MAX_STOCK = 20
PRICES = [7.0, 10.0, 13.0]  # price menu, same at every clock
START_STOCK = 15


def day_demand_mean(price, day):
    """Demand decays over the season (fad) and responds to price."""
    fad = 1.0 - 0.5 * day / DAYS
    return LAM_BASE * fad * (PRICE_REF / price) ** ELAST


def _trunc_poisson(lam, kmax=80):
    k = np.arange(kmax + 1)
    pmf = poisson.pmf(k, lam)
    pmf[-1] += 1 - pmf.sum()
    return pmf


def solve_clock(epochs, stock_max=MAX_STOCK):
    """Backward induction on (t, stock). Epoch demand at price p =
    Poisson(sum of that epoch's daily means) — exact aggregation of the
    daily process (Poisson additivity)."""
    days_per = DAYS // epochs
    pmf = {(p, t): _trunc_poisson(
               sum(day_demand_mean(p, d)
                   for d in range(t * days_per, (t + 1) * days_per)))
           for p in PRICES for t in range(epochs)}
    V = np.zeros((epochs + 1, stock_max + 1))       # V[epochs, :] = 0
    pol = np.zeros((epochs, stock_max + 1), dtype=int)
    for t in range(epochs - 1, -1, -1):
        for s in range(stock_max + 1):
            best_v, best_a = -1e18, 0
            for a, p in enumerate(PRICES):
                ev = 0.0
                for d_sold, pr in enumerate(pmf[(p, t)]):
                    sold = min(s, d_sold)
                    leftover = s - sold
                    tail = V[t + 1, leftover] if t + 1 < epochs else 0.0
                    # hold accrues WHILE stock waits: average inventory
                    # (s + leftover)/2 over the epoch, same convention
                    # at every clock (leftover alone undercharges coarse)
                    hold = HOLD_PER_DAY * days_per * (s + leftover) / 2
                    ev += pr * ((p - COST) * sold - hold + tail)
                if ev > best_v:
                    best_v, best_a = ev, a
            V[t, s], pol[t, s] = best_v, best_a
    return V, pol, (epochs + 1) * (stock_max + 1)


def main():
    print("Time granularity — same season, three clocks "
          f"({DAYS} days, fad decay, elasticity {ELAST})\n")
    vals = {}
    for name, epochs in (("weekly", 6), ("2-day", 12), ("daily", 42)):
        t0 = time.perf_counter()
        V, pol, n_states = solve_clock(epochs)
        dt = time.perf_counter() - t0
        vals[name] = V[0, START_STOCK]
        print(f"{name:7s} epochs={epochs:3d}  |S|={n_states:5d}  "
              f"V(s={START_STOCK})={V[0, START_STOCK]:8.2f}  "
              f"solve {dt*1000:6.1f} ms")
    print("\nvalue ordering check (must be non-decreasing in fineness):")
    ok = vals["weekly"] <= vals["2-day"] + 1e-9 <= vals["daily"] + 2e-9
    print(f"  weekly {vals['weekly']:.2f} <= 2-day {vals['2-day']:.2f} "
          f"<= daily {vals['daily']:.2f}  -> {'OK' if ok else 'VIOLATION'}")
    print("\nread:")
    print(" - a finer clock strictly enlarges the policy class -> value can")
    print("   only improve; the price is |S| growth (linear here) and more")
    print("   decisions to commit to.")
    print(" - the weekly plan cannot reprice after a demand shock inside the")
    print("   week; the daily plan pays for that option whether or not it")
    print("   ever fires (option premium vs information value).")
    print(" - rule of thumb: refine the clock until the marginal value of the")
    print("   last refinement stops paying for its |S|/|A| cost.")


if __name__ == "__main__":
    main()
