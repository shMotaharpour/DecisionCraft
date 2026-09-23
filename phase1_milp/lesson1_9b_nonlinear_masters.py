"""Lesson 1.9b — Nonlinear Masters: Column Generation beyond the LP Master.

Three live investigations, same course style (every claim measured):

(a) CONVEX master — congestion pricing. Blocks pick plans; total machine
    hours P incur a convex congestion cost (rho/2)P^2 on top of linear
    plan costs. The master is a convex QP; its "duals" come from KKT,
    not LP duality. CG = per-block pricing with gradient-adjusted costs,
    verified against FULL ENUMERATION of all plan pairs.

(b) CONCAVE master — the secant trap. Capacity purchase with a fixed
    charge f(k) = F*1[k>=1] + c*k is concave per unit. LP over LEVEL
    columns allows fractional convex combos: the reachable cost at
    demand d is the LOWER CONVEX HULL of the knots — for concave f the
    chords lie ABOVE f, so the LP "price" is NOT the true marginal cost.
    Measure: LP-CG cost vs integer optimum vs naive rounding, at
    integer and fractional demand.

(c) LAGRANGIAN relaxation — the coupling that breaks block separability.
    Two knapsack blocks share one machine capacity; the coupling row
    destroys block-diagonality so DW does not apply. Relax the coupling
    with multiplier mu: each block solves its OWN knapsack; subgradient
    ascent updates mu. Measure the Lagrangian bound vs direct MILP and
    the duality gap, plus how the bound converges.

Run: python phase1_milp/lesson1_9b_nonlinear_masters.py   (~10 s)
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

rng = np.random.default_rng(19)

# ======================================================================
# (a) CONVEX MASTER — congestion pricing via KKT, verified by enumeration
# ======================================================================
def part_a():
    print("=" * 68)
    print("(a) CONVEX MASTER — congestion cost (rho/2)P^2, KKT pricing")
    print("=" * 68)
    P_PLANS, B = 25, 2                      # plans per block, blocks
    cost = rng.integers(20, 80, size=(B, P_PLANS)).astype(float)
    mach = rng.integers(2, 9, size=(B, P_PLANS)).astype(float)
    rho = 1.4

    def pair_obj(p1, p2):
        return cost[0, p1] + cost[1, p2] + (rho / 2) * (mach[0, p1] + mach[1, p2]) ** 2

    # ground truth: enumerate all pairs
    best = min(((pair_obj(p1, p2), p1, p2)
                for p1 in range(P_PLANS) for p2 in range(P_PLANS)))
    true_opt, tp1, tp2 = best
    print(f"enumeration optimum: {true_opt:.4f}  plans=({tp1},{tp2})  "
          f"P={mach[0,tp1]+mach[1,tp2]:.0f}")

    # CG over P (total hours): for FIXED P, each block independently picks
    # argmin_p c_p + rho*P*m_p  -> F(P) = min over achievable P of pair cost
    # is the 1-D convex envelope; minimize F over P by golden section.
    from scipy.optimize import minimize_scalar

    def F(P):
        i1 = int(np.argmin(cost[0] + rho * P * mach[0]))
        i2 = int(np.argmin(cost[1] + rho * P * mach[1]))
        return pair_obj(i1, i2), i1, i2

    res = minimize_scalar(lambda P: F(P)[0], bounds=(2, 16),
                          method="bounded",
                          options={"xatol": 1e-10})
    P_star = res.x
    v, i1, i2 = F(P_star)
    # the envelope F(P) is piecewise (each block jumps plans as P grows);
    # ANY minimizer of the envelope is an optimal P — report the pair's own
    # P too so the printout is unambiguous
    v4, i14, i24 = F(float(mach[0, tp1] + mach[1, tp2]))
    print(f"CG (1-D envelope + per-block KKT pricing): {v:.4f}  "
          f"plans=({i1},{i2})  P*={P_star:.4f}")
    print(f"  envelope at the enumeration pair's P: {v4:.4f} "
          f"plans=({i14},{i24}) — same plateau")
    # KKT check: at the optimum, P must satisfy the stationarity of the
    # inner min — block pricing gradients must balance the envelope slope
    gap = abs(v - true_opt)
    print(f"|CG - enumeration| = {gap:.2e}")
    assert gap < 1e-6, "convex-master CG must match enumeration"

    # KKT story, made numeric: per-block reduced costs at P*
    rc0 = cost[0] + rho * P_star * mach[0] - (cost[0, i1] + rho * P_star * mach[0, i1])
    rc1 = cost[1] + rho * P_star * mach[1] - (cost[1, i2] + rho * P_star * mach[1, i2])
    print(f"KKT: min reduced cost per block = {rc0.min():.2e}, "
          f"{rc1.min():.2e} (both >= 0, zero at chosen plans)")
    print("read: the LP duals are gone; the pricing costs carry the")
    print("GRADIENT of the congestion term, and the convergence proof is")
    print("KKT (zero reduced costs), not LP duality. Exactness survives.\n")


# ======================================================================
# (b) CONCAVE MASTER — fixed charge, level columns, the secant trap
# ======================================================================
def part_b():
    print("=" * 68)
    print("(b) CONCAVE MASTER — fixed charge f(k)=F*1[k>0]+c*k, level columns")
    print("=" * 68)
    F_FIXED, C_VAR, K = 30.0, 4.0, 20

    def f(k):
        return 0.0 if k == 0 else F_FIXED + C_VAR * k

    for demand in (10.0, 10.5):
        # integer optimum: cheapest k >= demand (single knot)
        k_int = int(np.ceil(demand))
        opt = f(k_int)
        # LP over level columns: min cost to reach demand via convex combo
        # of knots 0..K (each column = knot j with cost f(j), "amount" j).
        # LP relaxation lower-bounds the MILP (fixed cost fractionality)
        # but the reachable cost sits on the SECANT between knots.
        from scipy.optimize import linprog
        # variables lambda_j (j=0..K): sum j*lam_j >= demand, sum lam = 1
        A_ub = np.vstack([-np.arange(K + 1), np.ones(K + 1)])
        b_ub = np.array([-demand, 1.0])
        c = np.array([f(j) for j in range(K + 1)])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None),
                      method="highs")
        lp_cost, lam = res.fun, res.x
        used = [(j, lam[j]) for j in range(K + 1) if lam[j] > 1e-9]
        # naive rounding: round every used level to its own integer (it IS
        # integer) — the real trap: the LP mix achieves demand fractionally;
        # the implementable single-knot plan is k_int at cost opt >= lp_cost
        frac_part = sum(x for j, x in used if abs(x - 1) > 1e-9) > 1e-9
        print(f"\ndemand={demand}: integer opt = f({k_int}) = {opt:.2f}")
        print(f"  LP-CG over level columns = {lp_cost:.2f}  mix={used}")
        print(f"  fractional mix used: {frac_part}")
        if lp_cost < opt - 1e-9:
            verdict = "LP underestimates — fixed-charge fractionality"
        else:
            verdict = "LP matches — knot demand"
        print(f"  LP vs integer: {100*(lp_cost-opt)/opt:+.2f}%  ({verdict})")
        # the secant reading: LP price of capacity between knots =
        # chord slope, i.e. the AVERAGE of the concave cost, NOT marginal
        print(f"  secant check: chord(0,{K}) slope = {f(K)/K:.2f}/unit vs "
              f"true marginal after fixed charge = {C_VAR:.2f}/unit")
    print("\nread: with a concave master, 'price = dual' breaks TWICE — the")
    print("fixed charge is fractionally shared in the LP (under-bidding the")
    print("true cost), and between knots the LP's implied per-unit price is")
    print("the secant, not the marginal. Remedies: keep it integer (B&P),")
    print("perspective reformulation (Frangioni-Gentile), or discretize the")
    print("usage levels into separate columns and pay the size.\n")


# ======================================================================
# (c) LAGRANGIAN RELAXATION — coupling that breaks block separability
# ======================================================================
def part_c():
    print("=" * 68)
    print("(c) LAGRANGIAN — shared-machine coupling, subgradient bound")
    print("=" * 68)
    NB, NJ, CAP = 2, 10, 14
    profit = rng.integers(5, 40, size=(NB, NJ)).astype(float)
    use = rng.integers(1, 6, size=(NB, NJ)).astype(float)

    # direct MILP (ground truth)
    nv = NB * NJ
    LOCAL = 4
    c_obj = -profit.flatten()
    # shared capacity + at most LOCAL jobs per block (block-local rows)
    A_rows = [use.flatten().reshape(1, -1),
              np.vstack([np.concatenate([np.ones(NJ), np.zeros(NJ)]),
                         np.concatenate([np.zeros(NJ), np.ones(NJ)])])]
    cons = [LinearConstraint(A_rows[0], -np.inf, CAP),
            LinearConstraint(A_rows[1], -np.inf, LOCAL * np.ones(2))]
    res = milp(c=c_obj, constraints=cons,
               integrality=np.ones(nv), bounds=Bounds(0, 1))
    milp_opt = -res.fun
    print(f"direct MILP optimum: {milp_opt:.1f} "
          f"(total use {float(A_rows[0].flatten() @ res.x):.1f} of cap {CAP})")

    # Lagrangian: relax the SHARED capacity row with mu >= 0 (block-local
    # rows stay in the subproblems, keeping them block-separable knapsacks)
    # L(mu) = mu*CAP + sum_b max{ (p - mu*a) x : 0<=x<=1, |x_b|<=LOCAL }

    def blocks_solve(mu):
        picks = []
        lag = mu * CAP
        for b in range(NB):
            red = profit[b] - mu * use[b]
            order = np.argsort(-red)
            chosen, load = [], 0
            for j in order:
                if red[j] > 0 and len(chosen) < LOCAL:
                    chosen.append(int(j))
                    load += 1
            picks.append(chosen)
            lag += float(red[chosen].sum())
        return lag, picks

    def total_use(picks):
        return sum(use[b, j] for b in range(NB) for j in picks[b])

    mu, best_bound, best_gap = 5.0, np.inf, None
    trace = []
    step0 = 1.5
    for it in range(300):
        lag, picks = blocks_solve(mu)
        g = CAP - total_use(picks)          # dL/dmu (subgradient)
        best_bound = min(best_bound, lag)   # dual: MINIMIZE the convex L(mu)
        # feasible repair: greedy by profit/use under the true capacity
        flat = [(profit[b, j] / use[b, j], b, j) for b in range(NB)
                for j in picks[b]]
        flat.sort(reverse=True)
        xu, tot_use, tot_p = set(), 0.0, 0.0
        for ratio, b, j in flat:
            if tot_use + use[b, j] <= CAP:
                xu.add((b, j)); tot_use += use[b, j]; tot_p += profit[b, j]
        best_gap = (milp_opt - tot_p) / milp_opt * 100
        trace.append((it, mu, lag, g))
        # subgradient DESCENT on the dual function; overuse (g<0) raises mu
        mu = max(0.0, mu - step0 * g / (1 + it * 0.03))
        if abs(g) < 1e-3 and it > 20:
            break
    print(f"Lagrangian dual best bound: {best_bound:.2f}  "
          f"(>= MILP by weak-duality theory)")
    print(f"duality gap vs MILP: {(best_bound - milp_opt) / milp_opt * 100:.2f}%")
    print(f"best feasible from repair: {tot_p:.1f} "
          f"-> optimality gap {best_gap:.2f}%")
    print(f"subgradient iterations: {len(trace)}  final mu={mu:.4f}")
    print("read: the coupling row broke block separability — DW cannot see")
    print("this structure. Lagrangian IS the master here: mu is the resource")
    print("price, each block prices its own jobs, and the bound quality is")
    print("measured, not assumed (gap printed above). This is exactly the")
    print("Chista opponent-model shape: price the shared machine, let blocks")
    print("decide locally.\n")


def main():
    part_a()
    part_b()
    part_c()
    print("summary table (see note): linear->LP duals; fixed-charge/integer")
    print("-> B&P; convex -> KKT pricing (verified here by enumeration);")
    print("concave -> level columns + integer master, LP price lies on the")
    print("secant (measured above); non-separable coupling -> Lagrangian")
    print("(bound + gap measured above).")


if __name__ == "__main__":
    main()
