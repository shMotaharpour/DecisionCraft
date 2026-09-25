"""Lesson 2.10a — Constraint vs penalty: hard cap, fixed mu, CMDP dual mu*.

Same inventory problem with a HARD BUDGET constraint: spend per order
(COST * order size) must stay under BUDGET. Three formulations:
  hard-constrained : budget-violating actions removed from A(s)
  penalty (fixed mu): r' = r - mu * overspend, solved unconstrained
  CMDP dual (mu*)  : the smallest penalty that makes the constraint bind
                     — by LP duality this IS the occupancy-LP shadow
                     price lambda* (lesson 2.7), found by bisection
Measures (honest, in the TRUE reward): mean discounted profit, fraction
of decisions violating the budget, budget utilization. Story: small mu
under-deters (constraint broken); large mu over-deters (value left on
the table); mu* prices the budget exactly — and is a readable business
number ("one more unit of budget/day is worth mu* profit").

Run: python phase2_mdp/lesson2_10a_constraint_penalty.py   (~20 s)
"""
import numpy as np
from scipy.stats import poisson

GAMMA = 0.95
MAX_S, MAX_A = 10, 10
PRICE, COST, HOLD, LAM = 10.0, 3.0, 0.4, 3.0
BUDGET = 6.0                # spend cap per order (COST * a <= 6 -> a <= 2)
PMF = poisson.pmf(np.arange(MAX_S + 1), LAM)
PMF[-1] += 1 - PMF.sum()


def reward(s, a, d):
    sold = min(s + a, d)
    return PRICE * sold - HOLD * max(s + a - d, 0) - COST * a


def build_P_R(pen_mu=0.0, allowed=None):
    P = np.zeros((MAX_S + 1, MAX_A + 1, MAX_S + 1))
    R = np.full((MAX_S + 1, MAX_A + 1), -1e9)
    for s in range(MAX_S + 1):
        for a in range(MAX_A - s + 1):
            if allowed is not None and not allowed[s, a]:
                continue
            R[s, a] = sum(pr * reward(s, a, d)
                          for d, pr in enumerate(PMF)) - pen_mu * max(
                              0.0, COST * a - BUDGET)
            for d, pr in enumerate(PMF):
                P[s, a, max(0, s + a - d)] += pr
    return P, R


def vi_policy(P, R):
    V = np.zeros(MAX_S + 1)
    pol = np.zeros(MAX_S + 1, dtype=int)
    for _ in range(500):
        Q = np.where(R > -1e8, R + GAMMA * np.einsum("sap,p->sa", P, V),
                     -1e18)
        Vn = Q.max(axis=1)
        pol = Q.argmax(axis=1)
        if np.abs(Vn - V).max() < 1e-10:
            break
        V = Vn
    return pol


def rollout(pol, seed=5, n=20000, enforce=False):
    """TRUE-reward rollouts. Returns mean profit, fraction of DECISIONS
    that violate the budget, mean order size."""
    rng = np.random.default_rng(seed)
    tot = 0.0
    n_dec = viol = spend_dec = 0
    for _ in range(n):
        s = int(rng.integers(0, MAX_S + 1))
        g, G = 1.0, 0.0
        for _ in range(30):
            a = min(int(pol[s]), MAX_S - s)
            if enforce and COST * a > BUDGET:
                a = int(min(BUDGET / COST, MAX_S - s))
            n_dec += 1
            if COST * a > BUDGET:
                viol += 1
            spend_dec += COST * a
            d = int(rng.choice(MAX_S + 1, p=PMF))
            G += g * reward(s, a, d)
            g *= GAMMA
            s = max(0, s + a - d)
        tot += G
    return tot / n, viol / n_dec, spend_dec / n_dec


def cmdp_dual_mu():
    """Bracket the smallest mu whose policy stops violating the budget.
    With discrete actions the violation rate JUMPS at the critical mu:
    below it, oversize orders pay; above, they don't. Bisection returns
    the bracketing interval's midpoint; the honest statement is that mu*
    lies in that interval (the CMDP shadow price, by LP duality)."""
    lo, hi = 0.0, 60.0        # lo violates, hi does not
    for _ in range(30):
        mid = (lo + hi) / 2
        pol = vi_policy(*build_P_R(pen_mu=mid))
        _, viol, _ = rollout(pol, n=20000)
        if viol > 0.005:
            lo = mid
        else:
            hi = mid
    return lo, hi


def main():
    print(f"Constraint vs penalty — spend cap {BUDGET} "
          f"(= {BUDGET / COST:.0f} units/order max)\n")

    allowed = np.zeros((MAX_S + 1, MAX_A + 1), bool)
    for s in range(MAX_S + 1):
        for a in range(MAX_A - s + 1):
            allowed[s, a] = COST * a <= BUDGET + 1e-9
    pol_hard = vi_policy(*build_P_R(allowed=allowed))
    perf_h, viol_h, spend_h = rollout(pol_hard, enforce=True)
    print(f"hard-constrained : profit={perf_h:7.2f}  violations=  0.00%  "
          f"avg spend={spend_h:.2f}")

    for mu in (1.0, 2.0, 5.0, 30.0):
        pol = vi_policy(*build_P_R(pen_mu=mu))
        perf, viol, spend = rollout(pol)
        tag = " (under-deters)" if viol > 0.001 else ""
        tag = tag or (" (over-deters)" if spend < spend_h - 0.05 else
                      " (binds)")
        print(f"penalty mu={mu:5.1f} : profit={perf:7.2f}  "
              f"violations={100 * viol:6.2f}%  avg spend={spend:.2f}{tag}")

    mu_lo, mu_hi = cmdp_dual_mu()
    # discrete-action landscape: violations jump from 41% (mu=2.0) to 0%
    # (mu=2.5). Scan a small grid inside the bracket for the honest mu*.
    grid = np.linspace(mu_lo, mu_hi, 6)
    mu_clean = mu_hi
    for mu in grid:
        pol_mu = vi_policy(*build_P_R(pen_mu=float(mu)))
        _, viol_mu, _ = rollout(pol_mu, n=20000)
        if viol_mu <= 0.005:
            mu_clean = float(mu)
            break
    pol_star = vi_policy(*build_P_R(pen_mu=mu_clean))
    perf_s, viol_s, spend_s = rollout(pol_star)
    print(f"CMDP dual mu* in [{mu_lo:.2f}, {mu_hi:.2f}] (clean at "
          f"{mu_clean:.2f}) : profit={perf_s:7.2f}  "
          f"violations={100 * viol_s:6.2f}%  avg spend={spend_s:.2f}")

    print("\nread:")
    print(" - mu=1/2 under-deter: the policy happily breaks the budget")
    print("   because the fine is cheaper than the profit it unlocks.")
    print(" - mu=30 over-deters risk is real: past the jump, the penalty")
    print("   policy collapses to the hard-cap solution (here they coincide")
    print("   because the discrete jump is sharp; with richer actions an")
    print("   over-priced mu buys LESS spend than the cap allows).")
    print(" - mu* is the constraint's shadow price (CMDP dual / LP duality):")
    print("   the smallest fine that makes the budget bind. With discrete")
    print("   actions it lands in a bracket, not a point — the violation")
    print("   rate JUMPS at the critical mu.")
    print(" - mu* is a readable business number: 'one more unit of budget")
    print("   per order is worth ~mu* per episode' — the same dual price")
    print("   that lessons 1.4 (MILP) and 4.6 (risk) put on constraints.")


if __name__ == "__main__":
    main()
