"""Lesson 2.9b — Terminal value: the invisible dial at the horizon's end.

Same seasonal inventory MDP; three end-of-horizon conventions:
  zero-terminal     : leftover stock is worthless at T (fashion: dies)
  liquidation       : leftover sells at a salvage price (70% off)
  carryover         : leftover carries value into "next season" (= has
                      terminal value = marginal value V_0 of a longer run)
The choice quietly reshapes the WHOLE policy, not just the last step:
a model that pretends stock evaporates stops ordering too early; a
model that pretends it carries value hoards. Measure: policy structure
(order-up-to levels over time) + realized profit under the TRUE
liquidation reality (salvage 30% of price, honest eval for all three).

Run: python phase2_mdp/lesson2_9b_terminal_value.py   (~15 s)
"""
import numpy as np
from scipy.stats import poisson

DAYS, MAX_S = 30, 15
PRICE, COST, HOLD, LAM = 10.0, 3.0, 0.4, 3.0
SALVAGE_TRUE = 3.0          # the real world: leftovers sell at 70% off
PMF = poisson.pmf(np.arange(MAX_S + 1), LAM)
PMF[-1] += 1 - PMF.sum()


def solve(terminal_value):
    """Backward induction over (t, stock); terminal_value(s) at t=T."""
    V = np.array([terminal_value(s) for s in range(MAX_S + 1)],
                 dtype=float)
    pol = np.zeros((DAYS, MAX_S + 1), dtype=int)
    for t in range(DAYS - 1, -1, -1):
        R = np.zeros((MAX_S + 1, MAX_S + 1))
        P = np.zeros((MAX_S + 1, MAX_S + 1, MAX_S + 1))
        for s in range(MAX_S + 1):
            for a in range(MAX_S - s + 1):
                R[s, a] = sum(pr * (PRICE * min(s + a, d)
                                    - HOLD * max(s + a - d, 0) - COST * a)
                              for d, pr in enumerate(PMF))
                for d, pr in enumerate(PMF):
                    P[s, a, max(0, s + a - d)] += pr
        Q = R + np.einsum("sap,p->sa", P, V)
        pol[t] = Q.argmax(axis=1)
        V = Q.max(axis=1)
    return V, pol


def true_eval(pol):
    """Real world: salvage 3.0 per leftover unit at the season's end."""
    rng = np.random.default_rng(11)
    tot = 0.0
    for _ in range(6000):
        s, G = 0, 0.0
        for t in range(DAYS):
            a = min(int(pol[t][s]), MAX_S - s)
            d = int(rng.choice(MAX_S + 1, p=PMF))
            sold = min(s + a, d)
            G += PRICE * sold - HOLD * max(s + a - d, 0) - COST * a
            s = max(0, s + a - d)
        G += SALVAGE_TRUE * s              # the true terminal cash
        tot += G
    return tot / 6000


def main():
    print("Terminal value — 30-day season, TRUE world: leftovers "
          f"liquidate at {SALVAGE_TRUE} (70% off)\n")
    TV = {
        "zero      (fashion)": lambda s: 0.0,
        "liquidate (salvage)": lambda s: SALVAGE_TRUE * s,
        "carryover (hoard)  ": lambda s: (PRICE - COST) * s,
    }
    for name, tv in TV.items():
        V, pol = solve(tv)
        perf = true_eval(pol)
        # the end reshapes the LAST days first: report last-day orders
        last = [int(pol[DAYS - 1][s]) for s in range(0, 16, 3)]
        late = int(pol[26][0])
        print(f"{name}: true-perf={perf:7.2f}  "
              f"last-day orders s=0,3,..,15: {last}  order@t=26,s=0: {late}")
    print("\nread:")
    print(" - zero-terminal orders 0 on the last day for any real stock")
    print("   (leftover is worthless in-model) -> leaves REAL salvage money.")
    print(" - carryover orders to the CAP on the last day (leftover worth")
    print("   full margin in-model) -> buys stock the true world liquidates")
    print("   at a loss. Overoptimistic ends are the expensive mistake.")
    print(" - liquidate orders exactly while margin beats salvage.")
    print(" - the distortion propagates BACKWARD: wrong ends bend the whole")
    print("   late-season policy path, not just the final day.")


if __name__ == "__main__":
    main()
