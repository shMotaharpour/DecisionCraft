"""Lesson 2.7 — From samples to values: MC & TD(0) policy evaluation,
plus the constrained-MDP bridge.

Part 1 — evaluate the (s,S) policy discovered in lesson 2.2
   (order 5-s when s<5, else 0) three ways:
   (a) EXACT: linear-system policy evaluation (lesson 2.2, ground truth)
   (b) Monte-Carlo returns (first-visit on one long trajectory): unbiased,
       noisy; verified against (a)
   (c) TD(0): online bootstrapping; biased early, converges with less data
Part 2 — CMDP bridge: same inventory with an ORDER-BUDGET constraint
   (average order units <= B). Occupancy-measure LP (proven equivalence,
   scipy.linprog); compare value/policy against the unconstrained optimum.
Run:  python lesson2_7_mc_td_cmdp.py
"""
import numpy as np
from scipy.optimize import linprog
from scipy.stats import poisson

MAX_INV = 10
LAMBDA = 3.0
PRICE, HOLD, COST = 10.0, 0.5, 3.0
GAMMA = 0.95
NS, NA_ = MAX_INV + 1, MAX_INV + 1

pmf = poisson.pmf(np.arange(MAX_INV + 1), LAMBDA)
pmf[-1] += 1.0 - pmf.sum()

# ---- model from lesson 2.2 + optimal policy via PI ------------------------
P = np.zeros((NS, NA_, NS))
R = np.full((NS, NA_), -1e9)
for s in range(NS):
    for a in range(NA_):
        if a > MAX_INV - s:
            continue
        R[s, a] = sum(pr * (PRICE * min(s + a, d) - HOLD * max(s + a - d, 0)
                            - COST * a) for d, pr in enumerate(pmf))
        for d, pr in enumerate(pmf):
            P[s, a, max(0, s + a - d)] += pr

policy = np.zeros(NS, dtype=int)
V_exact = None
for _ in range(100):
    P_pi = P[np.arange(NS), policy]
    R_pi = R[np.arange(NS), policy]
    V_exact = np.linalg.solve(np.eye(NS) - GAMMA * P_pi, R_pi)
    new_pol = (R + GAMMA * np.einsum("sap,p->sa", P, V_exact)).argmax(axis=1)
    if np.array_equal(new_pol, policy):
        break
    policy = new_pol
print(f"(a) PI optimal policy (order): {list(policy)}  V(s=0) = {V_exact[0]:.2f}")

P_pi = P[np.arange(NS), policy]
R_pi = R[np.arange(NS), policy]

# ---- (b) Monte-Carlo: first-visit returns on one long trajectory ----------
rng = np.random.default_rng(7)

def mc_eval(steps=2_000_000, thin=1000):
    traj_s = np.empty(steps, dtype=int)
    traj_r = np.empty(steps)
    s = 0
    for t in range(steps):
        traj_s[t] = s
        traj_r[t] = R_pi[s]
        s = rng.choice(NS, p=P_pi[s])
    G = np.empty(steps)                       # G_t = r_t + gamma G_{t+1}
    G[-1] = traj_r[-1]
    for t in range(steps - 2, -1, -1):
        G[t] = traj_r[t] + GAMMA * G[t + 1]
    sums = np.zeros(NS)
    counts = np.zeros(NS)
    for st in range(NS):
        idxs = np.flatnonzero(traj_s == st)   # every-visit indices
        sel = idxs[::thin]                    # thin to break autocorrelation
        sums[st] = G[sel].sum()
        counts[st] = len(sel)
    return sums / np.maximum(counts, 1), counts

V_mc, cnt = mc_eval()
print(f"(b) MC  (2M steps, thinned): computed — see accuracy report below")

# ---- (c) TD(0) -------------------------------------------------------------
def td0(alpha=0.02, steps=400_000):
    V = np.zeros(NS)
    s = 0
    for _ in range(steps):
        ns = rng.choice(NS, p=P_pi[s])
        V[s] += alpha * (R_pi[s] + GAMMA * V[ns] - V[s])
        s = ns
    return V

V_td = td0()
# under the (s,S) policy states 6..10 are never visited (s<=4 always orders
# up to 5, and demand never pushes stock above 5) — compare on the support
support = np.array([s for s in range(NS) if cnt[s] > 0])
err_mc = np.abs(V_mc - V_exact)[support]
err_td = np.abs(V_td - V_exact)[support]
print(f"(b) MC  (2M steps, thinned): max|V_mc - V_exact| on visited states = "
      f"{err_mc.max():.3f}")
print(f"(c) TD0 (400k steps, a=.02): max|V_td - V_exact| on visited states = "
      f"{err_td.max():.3f}")
print("    NOTE: states 6..10 have ZERO visits under this policy —")
print("    on-policy evaluation only learns where the policy goes. That is")
print("    exactly why phase 3 needs off-policy methods (Q-learning).")
print("    MC: unbiased, high variance (full-return noise).")
print("    TD: bootstrapped (biased while V is wrong), far lower variance.")
print("    This bias/variance trade is the seed of everything in phase 3.")

# ---- Part 2: CMDP via occupancy-measure LP ---------------------------------
print("\n--- CMDP: order-budget constraint (avg order units <= B) ---")
BUDGET = 2.0                                     # avg units per step
alpha0 = np.zeros(NS)
alpha0[0] = 1.0                                    # start empty
xs = NS * NA_
c = -R.flatten()                                   # maximize reward
A_eq, b_eq = [], []
for s in range(NS):                                # flow conservation
    row = np.zeros(xs)
    for a in range(NA_):
        row[s * NA_ + a] -= 1.0
        for sp in range(NS):
            row[sp * NA_ + a] += GAMMA * P[sp, a, s]
    A_eq.append(row)
    # NOTE sign: with the row as built (outflow negative, inflow positive),
    # conservation reads outflow - inflow = -alpha0[s]
    b_eq.append(-alpha0[s])
bud_row = np.zeros(xs)                             # discounted order units
for s in range(NS):
    for a in range(NA_):
        bud_row[s * NA_ + a] = a
res = linprog(c, A_ub=bud_row[None, :],
              b_ub=np.array([BUDGET]),
              A_eq=np.array(A_eq), b_eq=np.array(b_eq),
              bounds=(0, None), method="highs")
print(f"LP status: {res.message.split('.')[0]}")
val_unc = V_exact[0]
if res.success:
    x = res.x.reshape(NS, NA_)
    val_cmdp = -res.fun
    # average-per-step budget usage: discounted usage * (1-gamma)
    used = sum(x[s, a] * a for s in range(NS) for a in range(NA_)) * (1 - GAMMA)
    pol = x.argmax(axis=1)
    print(f"    unconstrained value (from s=0): {val_unc:.2f}")
    print(f"    budget-constrained value      : {val_cmdp:.2f}  "
          f"(avg order used {used:.2f} <= {BUDGET})")
    print(f"    cost of the constraint        : {val_unc - val_cmdp:.2f}")
    print(f"    constrained policy (order)    : {list(pol)}")
    print("    the LP rations orders where the marginal value per unit is")
    print("    highest — shadow-price logic from lesson 1, over state space.")
