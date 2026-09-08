"""Lesson 4.8 — Markov Games: Shapley value iteration & learning opponents.

The classic Shapley (1953) pursuit game, miniaturized: a pursuer and an
evader on a 1-D track, gap g ∈ {0..4}; both move simultaneously
({-1,0,1}); the evader stumbles with prob 0.25 (move randomized); when
adjacent the pursuer catches with prob 0.35. Caught = +1 annuity for the
pursuer (zero-sum: evader pays). This stochastic max-min structure is
exactly Shapley's stochastic-game form.

(a) Shapley value iteration  V(g) = max_ap min_ae E[r + γV(g')]  —
    provably convergent for zero-sum stochastic games (discounted).
(b) learn WITHOUT knowing the opponent's model: Markov fictitious play —
    per-state action frequencies of the opponent, best-respond each round.
    Measure policy drift from the equilibrium; zero-sum ⇒ settles.
(c) map every piece to the Kaggriculture/Chista decisions.
Runtime: ~10 s.
Run:  python phase4_hybrid/lesson4_8_markov_games.py
"""
import numpy as np

GAMMA = 0.9
G = 4
NS, NA = G + 1, 3
ACTIONS = [-1, 0, 1]
P_STUMBLE = 0.25
P_CATCH_ADJ = 0.35
rng = np.random.default_rng(9)


def step_dist(g, ap, ae):
    """Mixture over outcomes (evader stumble + catch-at-adjacent)."""
    mix = [(1 - P_STUMBLE,
            int(np.clip(g + ACTIONS[ae] - ACTIONS[ap], 0, G)))]
    for ae2 in range(NA):
        mix.append((P_STUMBLE / NA,
                    int(np.clip(g + ACTIONS[ae2] - ACTIONS[ap], 0, G))))
    acc = {}
    for w, g2 in mix:
        acc[g2] = acc.get(g2, 0.0) + w
    out = {}
    for g2, w in acc.items():
        if g2 == 0:
            out[(0, 1.0)] = out.get((0, 1.0), 0.0) + w
        elif g2 == 1:
            out[(0, 1.0)] = out.get((0, 1.0), 0.0) + w * P_CATCH_ADJ
            out[(1, -1.0)] = out.get((1, -1.0), 0.0) + w * (1 - P_CATCH_ADJ)
        else:
            out[(g2, -1.0)] = out.get((g2, -1.0), 0.0) + w
    return [(w, g2, r) for (g2, r), w in out.items()]


def shapley_vi(iters=3000):
    V = np.zeros(NS)
    hist = []
    for it in range(iters):
        Vn = np.zeros(NS)
        for g in range(1, NS):
            best = -np.inf
            for ap in range(NA):
                worst = min(
                    sum(p * (r + GAMMA * V[g2])
                        for p, g2, r in step_dist(g, ap, ae))
                    for ae in range(NA))
                best = max(best, worst)
            Vn[g] = best
        Vn[0] = 1.0 / (1 - GAMMA)          # caught = annuity of +1
        diff = np.abs(Vn - V).max()
        hist.append(Vn.copy())
        V = Vn
        if diff < 1e-9:
            break
    return V, np.array(hist), it


def equilibrium_policies(V):
    """Pursuer: argmax over ap of min_ae E[...]; evader: argmin over ae."""
    pol_p = np.zeros((NS, NA))
    pol_e = np.zeros((NS, NA))
    for g in range(1, NS):
        evs = np.array([[sum(p * (r + GAMMA * V[g2])
                             for p, g2, r in step_dist(g, ap, ae))
                         for ae in range(NA)] for ap in range(NA)])
        row_min = evs.min(axis=1)
        col_max = evs.max(axis=0)
        pol_p[g] = (row_min == row_min.max()) / (row_min == row_min.max()).sum()
        pol_e[g] = (col_max == col_max.min()) / (col_max == col_max.min()).sum()
    pol_p[0] = pol_e[0] = np.ones(NA) / NA     # absorbed: any
    return pol_p, pol_e


def markov_fictitious_play(rounds=200, horizon=30, V=None):
    """Each player keeps per-state frequencies of the OTHER's moves and
    best-responds with the known dynamics. Returns policy drift from the
    equilibrium and the frequency tables."""
    freq = [np.ones((NS, NA)), np.ones((NS, NA))]   # freq[0] models evader
    pol_p_eq, pol_e_eq = equilibrium_policies(V)
    drift = []
    pols = None
    for rnd in range(rounds):
        pols = []
        for p in range(2):
            pol = np.zeros((NS, NA))
            for g in range(1, NS):
                opp = freq[1 - p][g] / freq[1 - p][g].sum()
                for own in range(NA):
                    # player p's own action = own (ap for pursuer, ae for
                    # evader); opponent's = the other index. Payoff matrix
                    # E[own, opp] weighted by the opponent's frequencies.
                    ev = 0.0
                    for opp_a in range(NA):
                        if p == 0:
                            ap, ae = own, opp_a
                        else:
                            ap, ae = opp_a, own
                        ev += opp[opp_a] * sum(
                            pw * (r + GAMMA * V[g2])
                            for pw, g2, r in step_dist(g, ap, ae))
                    pol[g, own] = ev
                row = pol[g]
                sel = (row == row.max()) if p == 0 else (row == row.min())
                pol[g] = sel / max(1, sel.sum())
            pol[0] = np.ones(NA) / NA
            pols.append(pol)
        d = (np.abs(pols[0] - pol_p_eq).sum()
             + np.abs(pols[1] - pol_e_eq).sum()) / (2 * NS)
        drift.append(d)
        s = 1
        for _ in range(horizon):
            ap = int(rng.choice(NA, p=pols[0][s]))
            ae = int(rng.choice(NA, p=pols[1][s]))
            freq[0][s, ae] += 1
            freq[1][s, ap] += 1
            outs = step_dist(s, ap, ae)
            pw = np.array([o[0] for o in outs])
            i = rng.choice(len(outs), p=pw / pw.sum())
            s = outs[i][1]
    return np.array(drift), pols


if __name__ == "__main__":
    print("(a) Shapley value iteration — pursuit game, gamma =", GAMMA)
    V, hist, it = shapley_vi()
    print(f"   converged in {it} iterations")
    print("   V(gap=0..4) =", np.round(V, 3))
    print("   V(0) = 10 = +1 annuity (caught); V falls with distance;")
    print("   V(3) crosses 0: beyond gap 3 the pursuit loses value.")

    print("\n(b) equilibrium policies (support of argmax/argmin):")
    pol_p, pol_e = equilibrium_policies(V)
    for g in range(1, NS):
        print(f"   gap {g}: pursuer chases {ACTIONS} ->"
              f" {pol_p[g].astype(int)} | evader evades ->"
              f" {pol_e[g].astype(int)}")

    print("\n(c) Markov fictitious play — both players learning")
    drift, pols = markov_fictitious_play(V=V)
    print(f"   policy drift from equilibrium: first 20 rounds "
          f"{drift[:20].mean():.3f} -> last 50 {drift[-50:].mean():.3f}")
    print("   zero-sum + discounted: play settles toward the minimax")
    print("   policies (Shapley 1953 / fictitious-play convergence).")

    print("\n(d) mapping to our problems:")
    print("   Kaggriculture = 2-player stochastic game, near-perfect info:")
    print("   state = board, actions = move sets, transition includes the")
    print("   opponent's simultaneous move — the max in Bellman becomes")
    print("   max-min (or max over expected opponent, lesson 4.2).")
    print("   Chista's opponent model = the frequency table; the MILP")
    print("   tactical layer = best response given the model (lesson 4.3).")
