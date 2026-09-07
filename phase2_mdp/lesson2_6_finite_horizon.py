"""Lesson 2.6 — Finite-horizon backward induction: seasonal markdown.

A fashion item has T=12 selling weeks; stock is perishable (worthless after).
State (t, s): weeks left, units on hand. Action: price from a ladder
{24, 20, 16, 12}; demand is Poisson whose mean depends on price
(price elasticity). Reward = revenue; unsold units at the end are worthless.

(a) backward induction V_t(s) = max_a [ r(s,a) + E V_{t+1}(s - sold) ]
(b) compare with the BEST STATIONARY (s,S)-style rule from lesson 2.2's
    infinite-horizon logic: a single price that maximizes long-run value —
    the finite-horizon policy front-loads high prices when time is plenty
    and dumps when time runs out (time-dependent policy!). Report the
    value gap between stationary and time-dependent.
Run:  python lesson2_6_finite_horizon.py
"""
import numpy as np
from scipy.stats import poisson

HORIZON = 12
MAX_STOCK = 15
PRICES = np.array([24, 20, 16, 12])
LAM = np.array([0.4, 0.9, 2.2, 5.0])       # demand mean per price
GAMMA = 1.0                                 # finite horizon: no discount


def demand_pmf(price_idx):
    pmf = poisson.pmf(np.arange(MAX_STOCK + 1), LAM[price_idx])
    pmf[-1] += 1.0 - pmf.sum()
    return pmf


PMFS = [demand_pmf(k) for k in range(len(PRICES))]

# ---------------------------------------------- (a) backward induction
V = np.zeros((HORIZON + 1, MAX_STOCK + 1))       # V[T, :] = 0 (worthless)
policy = np.zeros((HORIZON, MAX_STOCK + 1), dtype=int)
for t in range(HORIZON - 1, -1, -1):
    for s in range(MAX_STOCK + 1):
        best_v, best_a = -1e9, 0
        for k, pmf in enumerate(PMFS):
            ev = sum(pr * (PRICES[k] * min(s, d) + V[t + 1][max(0, s - d)])
                     for d, pr in enumerate(pmf))
            if ev > best_v:
                best_v, best_a = ev, k
        V[t][s] = best_v
        policy[t][s] = best_a

print("A. backward induction over (weeks left t, stock s): optimal price")
print("   (P24 P20 P16 P12; '·' = stock exhausted)")
for t in range(HORIZON):
    row = " ".join("· " if s == 0 else f"P{PRICES[policy[t][s]]:<2d}"
                   for s in range(MAX_STOCK + 1))
    print(f"   t={t:2d}  {row}")
print(f"   V_0(s=15) = {V[0][MAX_STOCK]:8.2f}")

# ---------------------------------------------- (b) best stationary price
print("\nB. best single (stationary) price policy for the same 12 weeks")
for k, pr in enumerate(PRICES):
    # greedy-myopic value: always price k (simulate expectation via the
    # same recursion but restricted to one action)
    Vf = np.zeros(HORIZON + 1)
    Vs = np.zeros((HORIZON + 1, MAX_STOCK + 1))
    for t in range(HORIZON - 1, -1, -1):
        for s in range(MAX_STOCK + 1):
            Vs[t][s] = sum(
                p * (PRICES[k] * min(s, d) + Vs[t + 1][max(0, s - d)])
                for d, p in enumerate(PMFS[k]))
    print(f"   always P{pr:<2d}: V_0(s=15) = {Vs[0][MAX_STOCK]:8.2f}")
best_stat = max(
    (sum(p * (PRICES[k] * min(MAX_STOCK, d)) for d, p in enumerate(PMFS[k])),
     k) for k in range(len(PRICES)))
# value of the restricted-to-k recursion:
stat_vals = []
for k in range(len(PRICES)):
    Vs = np.zeros((HORIZON + 1, MAX_STOCK + 1))
    for t in range(HORIZON - 1, -1, -1):
        for s in range(MAX_STOCK + 1):
            Vs[t][s] = sum(p * (PRICES[k] * min(s, d) + Vs[t + 1][max(0, s - d)])
                           for d, p in enumerate(PMFS[k]))
    stat_vals.append(Vs[0][MAX_STOCK])
best_k = int(np.argmax(stat_vals))
gap = V[0][MAX_STOCK] - stat_vals[best_k]
print(f"   best stationary: P{PRICES[best_k]} at {stat_vals[best_k]:.2f}")
print(f"   time-dependent optimal: {V[0][MAX_STOCK]:.2f}")
print(f"   VALUE OF TIME-DEPENDENCE = {gap:.2f} "
      f"({100 * gap / stat_vals[best_k]:.1f}% better)")

print("\n   the time-dependent policy's price path from s=15:")
s = MAX_STOCK
path = []
for t in range(HORIZON):
    k = policy[t][s]
    # expected sales at that price
    e_sold = sum(d * p for d, p in enumerate(PMFS[k]))
    path.append(f"t{t}: P{PRICES[k]}(~{e_sold:.1f} u)")
    s = max(0, s - round(e_sold))
print("   " + " -> ".join(path))
