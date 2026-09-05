"""Lesson 2.4 — Beyond tabular: ALP approximation + MDP-as-LP + regime-augmented MDP.

(a) exact tabular VI on an inventory MDP with a market-regime dimension
    (bull/bear changes demand mean): shows where tabular still works.
(b) Approximate Linear Program: V(s) = theta @ phi(s) with 4 RBF features,
    solved with scipy.optimize.linprog; compare against exact V.
(c) the dual meaning: the ALP's dual gives state-pricing weights alpha.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.stats import poisson

MAX_INV = 10
GAMMA = 0.95
REGIMES = ["bull", "bear"]
LAM = {"bull": 4.0, "bear": 1.5}
REGIME_P = np.array([0.7, 0.3])  # each day: regime sampled i.i.d. (keeps it Markov)
PRICE, HOLD, COST = 10.0, 0.5, 3.0
NS, NA = MAX_INV + 1, MAX_INV + 1
NREG = len(REGIMES)

# states are (inv, regime) flattened: s = inv * NREG + regime
pmfs = {}
for r, lam in LAM.items():
    pmf = poisson.pmf(np.arange(MAX_INV + 1), lam)
    pmf[-1] += 1.0 - pmf.sum()
    pmfs[r] = pmf


def r_regime(inv, order, regime):
    pmf = pmfs[REGIMES[regime]]
    tot = 0.0
    for d, pr in enumerate(pmf):
        sold, left = min(inv + order, d), max(inv + order - d, 0)
        tot += pr * (PRICE * sold - HOLD * left - COST * order)
    return tot


def s2i(inv, reg):
    """state index: NOTE layout is s = reg*(MAX_INV+1) + inv so that divmod
    recovers (reg, inv) cleanly."""
    return reg * (MAX_INV + 1) + inv


NS_all = (MAX_INV + 1) * NREG
P = np.zeros((NS_all, NA, NS_all))
R = np.full((NS_all, NA), -1e9)
for inv in range(MAX_INV + 1):
    for reg in range(NREG):
        s = s2i(inv, reg)
        for a in range(NA):
            if a > MAX_INV - inv:
                continue
            R[s, a] = r_regime(inv, a, reg)
            pmf = pmfs[REGIMES[reg]]
            # regime re-sampled each day (i.i.d.) — the only stochastic row
            for d, pr in enumerate(pmf):
                ni = max(0, inv + a - d)
                for nr, pr_reg in enumerate(REGIME_P):
                    P[s, a, s2i(ni, nr)] += pr * pr_reg


def value_iteration(P, R, gamma):
    V = np.zeros(P.shape[0])
    for _ in range(50_000):
        Q = R + gamma * np.einsum("sap,p->sa", P, V)
        Vn = Q.max(axis=1)
        if np.abs(Vn - V).max() < 1e-10:
            return Vn
        V = Vn
    return V


print("(a) exact tabular VI on the regime-augmented MDP")
V = value_iteration(P, R, GAMMA)
Q = R + GAMMA * np.einsum("sap,p->sa", P, V)
print(f"    |S| = {NS_all} states; V(inv=0, bull) = {V[s2i(0,0)]:8.2f}   V(inv=0, bear) = {V[s2i(0,1)]:8.2f}")
pol = Q.argmax(axis=1)
print("    optimal target level per (inv, regime):")
for inv in range(0, MAX_INV + 1, 2):
    row = "  ".join(f"{REGIMES[r]}:{inv+pol[s2i(inv,r)]}" for r in range(NREG))
    print(f"    inv={inv:2d} -> {row}")

# ---------------------------------------------------------- (b) ALP
print("\n(b) Approximate Linear Program with 6 features")
def phi(s):
    """features over (inv, regime): piecewise-linear in inv + regime effects.
    Richer than 4-RBF because the true V* is nearly piecewise linear."""
    reg, inv = divmod(s, MAX_INV + 1)
    z = inv / MAX_INV
    feats = [1.0, z, z**2, 1.0 if reg == 0 else 0.0,
             z * (1.0 if reg == 0 else 0.0), max(0.0, z - 0.5)]
    return np.array(feats)

K = 6
# LP over theta (K vars). Constraints: for all s,a (valid):
#   theta.phi(s) >= R(s,a) + gamma * sum_s' P(s',a|s) theta.phi(s')
# objective: minimize sum_s alpha(s) theta.phi(s), alpha uniform
rows, rhs = [], []
valid = [(s, a) for s in range(NS_all) for a in range(NA) if R[s, a] > -1e8]
for s, a in valid:
    coef = phi(s) - GAMMA * np.einsum("p,sap->p", np.eye(K) @ np.array([phi(sp) for sp in range(NS_all)]).T, P[s, a]) if False else None

# build constraint matrix efficiently: phi_all[s] precomputed
PHI = np.array([phi(s) for s in range(NS_all)])          # [s, K]
for s, a in valid:
    succ = P[s, a]                                        # [s']
    exp_phi = succ @ PHI                                  # E[phi(s')|s,a]
    rows.append(PHI[s] - GAMMA * exp_phi)
    rhs.append(R[s, a])
A_ub = -np.array(rows)   # -(theta.phi(s) - gamma E) <= -R
b_ub = -np.array(rhs)
alpha = np.ones(NS_all) / NS_all
c_obj = alpha @ PHI                                       # minimize alpha.phi theta

# The true V* is not in the span of 4 features, so the pure Bellman
# inequality system can be INFEASIBLE (this actually happened on first
# run!). The standard fix: per-constraint slack variables with a big
# penalty in the objective — the ALP then minimizes total violation,
# i.e. projects V* onto the feature span in a Bellman-aware way.
n_cuts = len(valid)
A_slack = np.hstack([A_ub, -np.eye(n_cuts)])          # allow violation
c_full = np.concatenate([c_obj, np.full(n_cuts, 1e4)])
res = linprog(c_full, A_ub=A_slack, b_ub=b_ub,
              bounds=[(None, None)] * K + [(0, None)] * n_cuts, method="highs")
if not res.success:
    raise RuntimeError(f"ALP LP failed: {res.message}")
theta = np.asarray(res.x).flatten()[:K]
total_slack = res.x[K:].sum()
V_alp = PHI @ theta
err = np.abs(V_alp - V).max()
print(f"    LP status: {res.message.split('.')[0]}  (total slack = {total_slack:.1f})")
print(f"    max |V_ALP - V_exact| over {NS_all} states = {err:.3f}  (rel. {err/np.abs(V).max():.1%})")

# ---------------------------------------------------------- (c) dual
print("\n(c) ALP dual = state-pricing weights (occupancy-style)")
print(f"    dual variables shape: {np.asarray(res.ineqlin.marginals).shape}")
marg = -np.asarray(res.ineqlin.marginals).flatten()
# marginals are per-cut (132 cuts = 132 (s,a) pairs), not per-state; map cut
# pressure back to its STATE: for cut (s,a) the Bellman row was built at s.
cut_states = np.array([s for s, a in valid])
state_pressure = np.zeros(NS_all)
np.add.at(state_pressure, cut_states, marg)
top = np.argsort(state_pressure)[::-1][:5]
# note: with slack variables nearly every cut binds at its penalty price,
# so the dual reading is dominated by the slack cost — a real (and useful)
# caution about slack-penalized ALPs.
print("    top-5 states by total dual pressure (caution: slack-penalized):")
for s in top[:5]:
    reg, inv = divmod(s, MAX_INV + 1)
    print(f"      inv={inv:2d} regime={REGIMES[reg]:4s}  dual={state_pressure[s]:.3f}")
