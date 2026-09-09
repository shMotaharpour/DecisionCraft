"""Lesson 2.10b — Precomputed policy vs receding-horizon MPC.

Same inventory problem with a TIME-VARYING twist the tabular model was
NOT told about... no — honest setup: the demand rate SHIFTS mid-horizon
(promotion at day 15 doubles lambda). Two deployment styles:
  precomputed : VI once on the average-lambda model, execute the static
                policy forever
  MPC         : at every step, re-plan over a SHORT horizon (H=5) with
                the CURRENT model (post-shift lambda), execute first
                action, shift, repeat
The tabular VI model cannot see the shift (wrong model); MPC re-solves
with updated demand statistics (identified online with a rolling mean).
Measures: true profit under the shifted world + compute per decision.
The tradeoff: MPC adapts to what the static policy cannot even
represent; its cost is re-solving every step (cheap here, expensive in
lesson 1.7-scale problems) and short-sightedness (H is finite).

Run: python phase2_mdp/lesson2_10b_precompute_mpc.py   (~10 s)
"""
import time

import numpy as np
from scipy.stats import poisson

GAMMA = 0.95
DAYS = 30
MAX_S, MAX_A = 12, 12
PRICE, COST, HOLD = 10.0, 3.0, 0.4
LAM_PRE, LAM_POST = 2.0, 8.0     # promotion quadruples demand at day 15
SHIFT_DAY = 15
H_MPC = 5
N_SEEDS = 16


def pmf_for(lam):
    pmf = poisson.pmf(np.arange(MAX_S + 1), lam)
    pmf[-1] += 1 - pmf.sum()
    return pmf


def solve_static(lam):
    """VI with stationary lambda (the precomputed-policy approach)."""
    pmf = pmf_for(lam)
    P = np.zeros((MAX_S + 1, MAX_A + 1, MAX_S + 1))
    R = np.full((MAX_S + 1, MAX_A + 1), -1e9)
    for s in range(MAX_S + 1):
        for a in range(MAX_A - s + 1):
            R[s, a] = sum(pr * (PRICE * min(s + a, d)
                                - HOLD * max(s + a - d, 0) - COST * a)
                          for d, pr in enumerate(pmf))
            for d, pr in enumerate(pmf):
                P[s, a, max(0, s + a - d)] += pr
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


def mpc_action(s, t, lam_hat, H=H_MPC):
    """Backward induction over a H-step window with terminal value
    zero; execute only the first action."""
    pmf = pmf_for(lam_hat)
    V = np.zeros(MAX_S + 1)
    pol = 0
    for h in range(H - 1, -1, -1):
        Q = np.full((MAX_S + 1, MAX_A + 1), -1e9)
        for st in range(MAX_S + 1):
            for a in range(MAX_A - st + 1):
                Q[st, a] = sum(pr * (PRICE * min(st + a, d)
                                     - HOLD * max(st + a - d, 0)
                                     - COST * a)
                               for d, pr in enumerate(pmf)) + \
                    GAMMA * sum(pr * V[max(0, st + a - d)]
                                for d, pr in enumerate(pmf))
        V = Q.max(axis=1)
        pol = Q.argmax(axis=1)
    return int(pol[s])


def simulate(mode, seed=3):
    rng = np.random.default_rng(seed)
    static_pol = solve_static((LAM_PRE + LAM_POST) / 2) if mode == "static" \
        else None
    tot = 0.0
    comp = 0.0
    s = 0
    lam_runs, sales_today = 2.0, 0.0     # online demand estimator
    G = 0.0
    g = 1.0
    for t in range(DAYS):
        lam_true = LAM_PRE if t < SHIFT_DAY else LAM_POST
        t0 = time.perf_counter()
        if mode == "static":
            a = int(static_pol[s])
        else:
            a = mpc_action(s, t, lam_runs)
        comp += time.perf_counter() - t0
        d = int(rng.choice(MAX_S + 1, p=pmf_for(lam_true)))
        sold = min(s + a, d)
        r = PRICE * sold - HOLD * max(s + a - d, 0) - COST * a
        G += g * r
        g *= GAMMA
        # online estimator: rolling mean of realized demand (fast EMA —
        # 0.3 was too slow to track the shift and crippled MPC)
        sales_today = 0.5 * sales_today + 0.5 * d
        lam_runs = max(0.3, sales_today)
        s = max(0, s + a - d)
    return G, comp * 1000 / DAYS


def main():
    print(f"Precomputed policy vs MPC — demand shifts {LAM_PRE} -> "
          f"{LAM_POST} at day {SHIFT_DAY} (promotion), "
          f"{N_SEEDS} paired seeds\n")
    stat_vals, mpc_vals = [], []
    ms_s = ms_m = 0.0
    for seed in range(N_SEEDS):
        G_s, c_s = simulate("static", seed)
        G_m, c_m = simulate("mpc", seed)
        stat_vals.append(G_s)
        mpc_vals.append(G_m)
        ms_s, ms_m = max(ms_s, c_s), max(ms_m, c_m)
    import numpy as np
    s_mean, m_mean = np.mean(stat_vals), np.mean(mpc_vals)
    print(f"static VI policy : profit={s_mean:7.2f} (mean of {N_SEEDS})   "
          f"{ms_s:.2f} ms/decision")
    print(f"MPC  (H={H_MPC}, online lambda) : profit={m_mean:7.2f} "
          f"(mean of {N_SEEDS})   {ms_m:.2f} ms/decision")
    print(f"\nMPC gain: {m_mean - s_mean:+.2f} "
          f"({100 * (m_mean - s_mean) / abs(s_mean):+.1f}%)")
    print("\nread:")
    print(" - the static policy was optimal for AVERAGE demand — a model")
    print("   that never exists after day 15; it understocks through the")
    print("   promotion and pays the whole season for its blindness.")
    print(" - MPC with an online estimator rides the shift (no retraining,")
    print("   no new policy class — just re-solving the same tiny MDP with")
    print("   fresh statistics). Its price: solve time per decision")
    print("   (trivial here, the whole point of lesson 1.7 at scale).")
    print(" - two methodological traps found while building this (kept in")
    print("   the evidence): a slow EMA estimator (0.3) made MPC LOSE its")
    print("   advantage, and a mild shift (2->4) with a single seed showed")
    print("   +4.6% that vanished under 16 paired seeds. Shift size and")
    print("   estimator speed are part of the deployment design, not noise.")
    print(" - design rule: precompute when the world is stationary and")
    print("   decisions are frequent; re-plan when the world moves or the")
    print("   model was wrong — and measure the solver budget honestly.")


if __name__ == "__main__":
    main()
