"""Lesson 2.1 — Markov chains from scratch (NumPy only).

Market regimes: bull / bear / stagnant.
(a) simulate 100,000 months, empirically recover the transition matrix
(b) k-step forecasts via matrix powers
(c) stationary distribution via eigen decomposition, verified against
    the simulation histogram
"""
import numpy as np

STATES = ["bull", "bear", "stagnant"]

P = np.array([
    [0.85, 0.10, 0.05],   # from bull
    [0.20, 0.70, 0.10],   # from bear
    [0.30, 0.30, 0.40],   # from stagnant
])
assert np.allclose(P.sum(axis=1), 1.0), "rows must sum to 1"

# ---------------------------------------------------------- (a) simulation
rng = np.random.default_rng(42)
N = 100_000
state = 1  # start in bear
visits = np.zeros(3, dtype=np.int64)
transitions = np.zeros((3, 3), dtype=np.int64)

for _ in range(N):
    visits[state] += 1
    nxt = rng.choice(3, p=P[state])
    transitions[state, nxt] += 1
    state = nxt

P_emp = transitions / transitions.sum(axis=1, keepdims=True)
print("(a) empirical transition matrix after 100k months:")
for i, row in enumerate(P_emp):
    print(f"    from {STATES[i]:9s}: " + "  ".join(f"{j:0.3f}" for j in row))
print("    max abs deviation from P:", f"{np.abs(P_emp - P).max():.4f}")

# ---------------------------------------------------------- (b) k-step
pi_bear = np.array([0.0, 1.0, 0.0])
for k in (1, 3, 12):
    forecast = pi_bear @ np.linalg.matrix_power(P, k)
    print(f"(b) start in bear, P(in bull) after {k:2d} months = {forecast[0]:.3f}")

# ---------------------------------------------------------- (c) stationary
evals, evecs = np.linalg.eig(P.T)
k = np.argmin(np.abs(evals - 1.0))
pi_star = np.real(evecs[:, k])
pi_star = pi_star / pi_star.sum()
print("(c) stationary distribution (analytic):")
for s, p in zip(STATES, pi_star):
    print(f"    {s:9s} = {p:.4f}")

print("    long-run share from simulation (visits/N):")
for s, v in zip(STATES, visits / N):
    print(f"    {s:9s} = {v:.4f}")

print("    convergence check |sim - analytic|:",
      f"{np.abs(visits / N - pi_star).max():.4f}")
print("    eigenvalues of P (memory decay):", np.round(np.sort_complex(evals), 3))
