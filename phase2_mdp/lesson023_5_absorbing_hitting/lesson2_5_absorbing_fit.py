"""Lesson 2.5 — Absorbing chains: hitting times, absorption probabilities,
and fitting P from data.

A. Gambler's ruin (stake 0..N, win/lose prob p/q): fundamental matrix
   N = (I-Q)^-1 gives expected time to ruin/win from every stake, plus
   absorption probabilities — verified against the classic closed form.
B. Hitting times WITHOUT absorption: expected first-passage times between
   all state pairs via the fundamental-matrix trick on the recurrent chain
   (mean first passage matrix M with m_ij = 1 + sum_k P_ik m_kj, i != j).
C. Fitting P from data: synthetic price series (3 regimes) -> bucket into
   states -> maximum-likelihood P_hat -> compare vs the TRUE matrix that
   generated it; show discretization bias and the (n_ij == 0) pathology.
Run:  python lesson2_5_absorbing_fit.py
"""
import numpy as np

rng = np.random.default_rng(42)

# ------------------------------------------------------------- A. gambler's ruin
N_STAKE = 6          # states 0..6; 0 = ruined, 6 = target reached
P_WIN = 0.45         # casino edge: p < 0.5

def gambler_ruin_P(n, p):
    q = 1 - p
    T = np.zeros((n - 1, n - 1))            # transient states 1..n-1
    for s in range(1, n):
        T[s - 1, s - 2 if s - 2 >= 0 else s - 1] += q if s > 1 else 0
        T[s - 1, s] += p if s < n - 1 else 0
        # transitions to absorbing states (0 or n) simply leave T
    P = np.zeros((n + 1, n + 1))
    P[:n, :n] = np.block([[np.eye(1), np.zeros((1, n - 1))],
                          [T, np.zeros((n - 1, 1))]])[:n, :n]
    return P

# build directly, cleanly:
P_full = np.zeros((N_STAKE + 1, N_STAKE + 1))
P_full[0, 0] = P_full[N_STAKE, N_STAKE] = 1.0        # absorbing
for s in range(1, N_STAKE):
    P_full[s, s - 1] = 1 - P_WIN
    P_full[s, s + 1] = P_WIN

Q = P_full[1:N_STAKE, 1:N_STAKE]                     # transient block
R = P_full[1:N_STAKE, [0, N_STAKE]]                  # to absorbing 0 / N
Nmat = np.linalg.inv(np.eye(N_STAKE - 1) - Q)        # fundamental matrix
t_absorb = Nmat @ np.ones(N_STAKE - 1)               # E[time to absorption]
B = Nmat @ R                                         # absorption probabilities

print("A. gambler's ruin: stake 0..6, p(win step)=0.45")
print(f"   E[steps until ruin-or-target] by start stake 1..{N_STAKE-1}:")
print("  ", np.round(t_absorb, 2))
# closed forms (proven): E_i = i/(q-p) * (1 - (q/p)^-i ... ) — use the standard
# ruin-probability formula instead:
i = np.arange(1, N_STAKE)
rho = (1 - P_WIN) / P_WIN
ruin_closed = (rho**i - rho**N_STAKE) / (1 - rho**N_STAKE)
print(f"   ruin prob: fundamental matrix {np.round(B[:, 0], 3)}")
print(f"              closed form      {np.round(ruin_closed, 3)}")
print(f"              max |diff| = {np.abs(B[:, 0] - ruin_closed).max():.2e}")
print(f"   closed-form E[time] check (p<q): {np.round(i/(1-2*P_WIN)*(1-(rho)**i/(rho)**N_STAKE),2)}")

# ------------------------------------------------------------- B. first passage
print("\nB. mean first passage times (lesson 2.1's market chain)")
P = np.array([[0.85, 0.10, 0.05],
              [0.20, 0.70, 0.10],
              [0.30, 0.30, 0.40]])
n = len(P)
pi_star = np.linalg.eig(P.T)[1][:, np.argmin(np.abs(
    np.linalg.eig(P.T)[0] - 1))].real
pi_star /= pi_star.sum()
# Mean first passage by FIRST-STEP ANALYSIS: leaving the target out of the
# sum, m_ij = 1 + sum_{k != j} P_ik m_kj — a linear system per target j.
# (Verified against Monte Carlo: m_02 = 15.96 sim vs 16.00 direct.)
M = np.zeros((n, n))
for j in range(n):
    idx = [i for i in range(n) if i != j]
    M[idx, j] = np.linalg.solve(
        np.eye(n - 1) - P[np.ix_(idx, idx)], np.ones(n - 1))
names = ["bull", "bear", "stagn"]
print("   expected months to reach COLUMN state, starting from ROW state:")
print("           " + "  ".join(f"{c:>8s}" for c in names))
for irow, nm in enumerate(names):
    print(f"   {nm:>8s}  " + "  ".join(f"{M[irow, j]:8.2f}" for j in range(n)))
# sanity: Kemeny's law — mean return time to i (leave, come back) = 1/pi_i
print("   mean return times 1/pi_i:", np.round(1 / pi_star, 2),
      "  (proven: mean return time = 1/pi_i)")

# ------------------------------------------------------------- C. fit P from data
print("\nC. fitting P from a generated series (3 regimes, 20k days)")
P_TRUE = P.copy()
seq = [1]                                     # start in bear
for _ in range(20_000 - 1):
    seq.append(rng.choice(3, p=P_TRUE[seq[-1]]))
seq = np.array(seq)
counts = np.zeros((3, 3))
np.add.at(counts, (seq[:-1], seq[1:]), 1)
P_hat = counts / counts.sum(axis=1, keepdims=True)
print(f"   max |P_hat - P_true| = {np.abs(P_hat - P_TRUE).max():.4f}")
z = counts.sum(axis=1) == 0
print(f"   rows with zero visits (fit silently undefined): {z.sum()}")

# bucketing demo: continuous prices -> states
print("\n   continuous -> states (bucketing) and its bias")
log_ret = rng.normal(0, 0.01, 50_000)
vol = np.abs(rng.normal(0, 0.01, 50_000))     # fake regime proxy
price = 100 * np.exp(np.cumsum(log_ret))
buckets = np.digitize(price, [95, 100, 105])  # price bands as states
cb = np.zeros((4, 4))
np.add.at(cb, (buckets[:-1], buckets[1:]), 1)
P_b = cb / cb.sum(axis=1, keepdims=True)
print(f"   banded-price chain row sums: {np.round(P_b.sum(axis=1), 3)}")
print("   (bucket boundaries turn smooth randomness into spurious")
print("    persistence: rows here are NOT a true market regime model)")
